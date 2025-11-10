import os
import time
import argparse
import pickle
import logging
from collections import deque, defaultdict
from typing import Deque, Dict, Any, List

import numpy as np
import pandas as pd
from web3 import Web3
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_artifacts(models_dir: str = "models"):
    model_path = Path(models_dir) / "mev_detector_improved.pkl"
    scaler_path = Path(models_dir) / "feature_scaler_improved.pkl"
    features_path = Path(models_dir) / "feature_columns_improved.txt"

    if not model_path.exists() or not scaler_path.exists() or not features_path.exists():
        raise FileNotFoundError("Model, scaler, or feature columns file not found in models/ directory")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'r') as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    logger.info(f"Loaded model ({model_path.name}), scaler and {len(feature_cols)} features")
    return model, scaler, feature_cols


def init_web3(ws_url: str) -> Web3:
    # web3.py v7 has different provider structure
    # Try WebsocketProviderV2 first, fall back to legacy
    try:
        from web3.providers.websocket import WebsocketProviderV2
        provider = WebsocketProviderV2(ws_url)
    except (ImportError, AttributeError):
        try:
            from web3.providers.legacy_websocket import LegacyWebSocketProvider
            provider = LegacyWebSocketProvider(ws_url)
        except ImportError:
            raise ImportError("No compatible WebSocket provider. Install: pip install websocket-client")
    
    w3 = Web3(provider)
    if not w3.is_connected():
        raise ConnectionError(f"Unable to connect to {ws_url}")
    logger.info("Connected to Web3 provider")
    return w3


def compute_features(tx: Dict[str, Any], buffer: Deque[Dict[str, Any]], lookback: int, feature_cols: List[str]) -> Dict[str, float]:
    """Compute a subset of features compatible with the trained model.
    buffer contains dicts with keys: hash, to, from, gasPrice, gas, value, nonce, ts
    """
    # Build arrays from buffer (most recent lookback transactions)
    recent = list(buffer)[-lookback:]
    gas_arr = np.array([t['gasPrice'] for t in recent if t['gasPrice'] is not None], dtype=float)
    to_arr = np.array([t['to'] or '0x0' for t in recent], dtype=object)
    from_arr = np.array([t['from'] for t in recent], dtype=object)
    nonce_arr = np.array([t['nonce'] if t['nonce'] is not None else -1 for t in recent], dtype=int)
    ts_arr = np.array([t['ts'] for t in recent], dtype=float)
    gas_limit_arr = np.array([t.get('gas') or 0 for t in recent], dtype=float)
    value_arr = np.array([t.get('value') or 0 for t in recent], dtype=float)

    cur_gas = float(tx.get('gasPrice') or 0)
    cur_to = tx.get('to') or '0x0'
    cur_from = tx.get('from')
    cur_nonce = tx.get('nonce') if tx.get('nonce') is not None else -1
    now_ts = tx.get('ts', time.time())

    feat = defaultdict(float)

    # Basic
    feat['gas_price'] = cur_gas
    feat['gas_limit'] = float(tx.get('gas') or 0)
    feat['tx_value'] = float(tx.get('value') or 0)

    # Recent gas stats
    if len(gas_arr) > 0:
        feat['recent_gas_mean'] = float(np.mean(gas_arr))
        feat['recent_gas_std'] = float(np.std(gas_arr))
        feat['recent_gas_median'] = float(np.median(gas_arr))
        feat['recent_gas_max'] = float(np.max(gas_arr))
        feat['recent_gas_min'] = float(np.min(gas_arr))
        feat['gas_vs_mean'] = cur_gas / (feat['recent_gas_mean'] + 1e-9)
        feat['gas_vs_median'] = cur_gas / (feat['recent_gas_median'] + 1e-9)
        feat['gas_vs_max'] = cur_gas / (feat['recent_gas_max'] + 1e-9)
        p75, p90, p95, p99 = np.percentile(gas_arr, [75, 90, 95, 99])
        feat['gas_percentile_75'] = float(p75)
        feat['gas_percentile_90'] = float(p90)
        feat['gas_percentile_95'] = float(p95)
        feat['is_top_10pct'] = 1.0 if cur_gas > p90 else 0.0
        feat['is_top_5pct'] = 1.0 if cur_gas > p95 else 0.0
        feat['is_top_1pct'] = 1.0 if cur_gas > p99 else 0.0
        feat['gas_volatility'] = float(feat['recent_gas_std'] / (feat['recent_gas_mean'] + 1e-9))
        feat['gas_range'] = float((feat['recent_gas_max'] - feat['recent_gas_min']) / (feat['recent_gas_mean'] + 1e-9))
    else:
        # defaults
        feat['recent_gas_mean'] = feat['recent_gas_std'] = feat['recent_gas_median'] = 0.0

    # Target-based
    same_target_mask = to_arr == cur_to if len(to_arr) > 0 else np.array([])
    if len(same_target_mask) > 0:
        same_count = int(same_target_mask.sum())
        feat['same_target_count'] = same_count
        feat['same_target_ratio'] = same_count / max(1, len(to_arr))
        target_gas_prices = gas_arr[same_target_mask]
        if len(target_gas_prices) > 0:
            feat['target_gas_max'] = float(np.max(target_gas_prices))
            feat['target_gas_mean'] = float(np.mean(target_gas_prices))
            feat['target_gas_min'] = float(np.min(target_gas_prices))
            feat['gas_vs_target_max'] = cur_gas / (feat['target_gas_max'] + 1e-9)
            feat['gas_vs_target_mean'] = cur_gas / (feat['target_gas_mean'] + 1e-9)
            feat['beating_target_max'] = 1.0 if cur_gas > feat['target_gas_max'] else 0.0
            if len(target_gas_prices) >= 2:
                diffs = np.diff(target_gas_prices)
                feat['target_gas_escalating'] = 1.0 if (diffs > 0).sum() > len(diffs) * 0.6 else 0.0
    else:
        feat['same_target_count'] = 0.0
        feat['same_target_ratio'] = 0.0

    # Sender-based
    sender_mask = from_arr == cur_from if len(from_arr) > 0 else np.array([])
    feat['sender_recent_count'] = int(sender_mask.sum()) if len(sender_mask) > 0 else 0
    # nonce replacement: same sender and same nonce seen earlier
    if len(from_arr) > 0:
        same_nonce_mask = (from_arr == cur_from) & (nonce_arr == cur_nonce)
        feat['sender_has_same_nonce'] = 1.0 if same_nonce_mask.sum() > 0 else 0.0
    else:
        feat['sender_has_same_nonce'] = 0.0

    # Temporal
    if len(ts_arr) > 1:
        feat['time_span_sec'] = float(now_ts - ts_arr[0])
        feat['tx_rate'] = float(len(ts_arr) / (max(1e-6, now_ts - ts_arr[0])))
    else:
        feat['time_span_sec'] = 0.0
        feat['tx_rate'] = 0.0

    # Fill missing expected features as zeros
    out = {k: float(feat.get(k, 0.0)) for k in feature_cols}
    return out


def generate_alert_reason(feat_dict: Dict[str, float], score: float) -> str:
    """Generate human-readable reason for MEV alert"""
    reasons = []
    
    # Nonce replacement (strongest signal)
    if feat_dict.get('sender_has_same_nonce', 0) > 0:
        reasons.append("Nonce replacement")
    
    # Same target activity
    same_target = int(feat_dict.get('same_target_count', 0))
    if same_target >= 3:
        reasons.append(f"{same_target} txs to same target")
    
    # Gas escalation
    if feat_dict.get('target_gas_escalating', 0) > 0:
        reasons.append("Gas escalating")
    
    # Beating previous gas prices
    if feat_dict.get('beating_target_max', 0) > 0:
        reasons.append("Outbidding previous txs")
    
    # High gas price
    if feat_dict.get('is_top_1pct', 0) > 0:
        reasons.append("Top 1% gas")
    elif feat_dict.get('is_top_5pct', 0) > 0:
        reasons.append("Top 5% gas")
    
    return " | ".join(reasons) if reasons else "Pattern detected"


def run_live_scoring(ws_url: str, threshold: float = 0.9, lookback: int = 50, models_dir: str = 'models'):
    model, scaler, feature_cols = load_artifacts(models_dir)
    w3 = init_web3(ws_url)

    # Sliding buffer of recent txs
    buffer: Deque[Dict[str, Any]] = deque(maxlen=1000)

    # We subscribe to pending transactions
    sub = w3.eth.filter('pending')
    
    # Stats tracking
    tx_count = 0
    alert_count = 0
    
    # Store last 60 transactions with scores for display
    recent_txs = deque(maxlen=60)
    
    console.print("[bold green]Live MEV Detector Started[/bold green]")
    console.print(f"Threshold: {threshold:.4f} | Lookback: {lookback}\n")

    def render_table():
        """Render the current state of recent transactions"""
        warmup_remaining = max(0, lookback - len(buffer))
        if warmup_remaining > 0:
            title = f"[bold]Live Mempool Feed[/bold] | Warming up... ({len(buffer)}/{lookback})"
        else:
            alert_pct = 100 * alert_count / max(1, tx_count)
            title = f"[bold]Live Mempool Feed[/bold] | Total: {tx_count} | Alerts: {alert_count} ({alert_pct:.2f}%)"
        
        # Get terminal width and calculate column widths dynamically
        term_width = console.width
        
        # Calculate available width (subtract borders and padding)
        available_width = term_width - 10  # Account for borders and spacing
        
        # Allocate widths proportionally
        hash_width = max(16, int(available_width * 0.20))
        from_width = max(10, int(available_width * 0.15))
        to_width = max(10, int(available_width * 0.15))
        gas_width = max(10, int(available_width * 0.12))
        score_width = max(8, int(available_width * 0.10))
        status_width = max(15, available_width - hash_width - from_width - to_width - gas_width - score_width - 8)
        
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED, title=title, expand=True)
        table.add_column("#", style="dim", width=6)
        table.add_column("Hash", width=hash_width)
        table.add_column("From", width=from_width)
        table.add_column("To", width=to_width)
        table.add_column("Gas (Gwei)", justify="right", width=gas_width)
        table.add_column("Score", justify="right", width=score_width)
        table.add_column("Status", width=status_width)
        
        for tx_data in reversed(recent_txs):  # Most recent at top
            tx_num = tx_data['num']
            tx_rec = tx_data['tx']
            score = tx_data['score']
            is_alert = tx_data['is_alert']
            reason = tx_data.get('reason', '')
            
            hash_short = tx_rec['hash'][:16] + "..."
            from_short = (tx_rec['from'][:10] + "..." if tx_rec['from'] else "N/A")
            to_short = tx_rec['to'][:10] + "..."
            gas_gwei = (tx_rec['gasPrice'] / 1e9) if tx_rec['gasPrice'] else 0
            
            if is_alert:
                table.add_row(
                    f"[bold red]{tx_num}[/bold red]",
                    f"[red]{hash_short}[/red]",
                    f"[red]{from_short}[/red]",
                    f"[red]{to_short}[/red]",
                    f"[bold red]{gas_gwei:.2f}[/bold red]",
                    f"[bold red]{score:.4f}[/bold red]",
                    f"[bold yellow]ALERT: {reason}[/bold yellow]"
                )
            elif score == 0.0:
                # Warming up
                table.add_row(
                    str(tx_num),
                    f"[dim]{hash_short}[/dim]",
                    f"[dim]{from_short}[/dim]",
                    f"[dim]{to_short}[/dim]",
                    f"{gas_gwei:.2f}",
                    "[dim]-[/dim]",
                    "[yellow]Warming...[/yellow]"
                )
            else:
                table.add_row(
                    str(tx_num),
                    f"[dim]{hash_short}[/dim]",
                    f"[dim]{from_short}[/dim]",
                    f"[dim]{to_short}[/dim]",
                    f"{gas_gwei:.2f}",
                    f"{score:.4f}",
                    "[green]OK[/green]"
                )
        
        return table

    try:
        with Live(render_table(), refresh_per_second=4, console=console) as live:
            while True:
                hashes = sub.get_new_entries()
                for h in hashes:
                    try:
                        tx = w3.eth.get_transaction(h)
                    except Exception as e:
                        continue

                    arrival_ts = time.time()
                    tx_record = {
                        'hash': tx['hash'].hex() if hasattr(tx['hash'], 'hex') else str(tx['hash']),
                        'to': tx['to'].lower() if tx['to'] else '0x0',
                        'from': tx['from'].lower() if 'from' in tx else (tx['sender'].lower() if 'sender' in tx else None),
                        'gasPrice': float(tx['gasPrice']) if 'gasPrice' in tx else None,
                        'gas': float(tx['gas']) if 'gas' in tx else 0.0,
                        'value': float(tx['value']) if 'value' in tx else 0.0,
                        'nonce': int(tx['nonce']) if 'nonce' in tx else -1,
                        'ts': arrival_ts
                    }

                    buffer.append(tx_record)
                    tx_count += 1

                    # Only start scoring once we have at least `lookback` items
                    if len(buffer) >= lookback:
                        feat_dict = compute_features(tx_record, buffer, lookback, feature_cols)
                        X = pd.DataFrame([feat_dict])
                        X_scaled = scaler.transform(X)

                        # LightGBM Booster predict gives probability
                        try:
                            score = float(model.predict(X_scaled)[0])
                        except Exception:
                            score = float(model.predict_proba(X_scaled)[:, 1][0])

                        is_alert = score >= threshold
                        if is_alert:
                            alert_count += 1
                            reason = generate_alert_reason(feat_dict, score)
                        else:
                            reason = ""
                        
                        # Add to recent transactions
                        tx_data = {
                            'num': tx_count,
                            'tx': tx_record,
                            'score': score,
                            'is_alert': is_alert,
                            'reason': reason,
                            'features': feat_dict
                        }
                        recent_txs.append(tx_data)
                    else:
                        # During warmup, still show transactions
                        tx_data = {
                            'num': tx_count,
                            'tx': tx_record,
                            'score': 0.0,
                            'is_alert': False,
                            'reason': '',
                            'features': {}
                        }
                        recent_txs.append(tx_data)
                    
                    # Update the live display
                    live.update(render_table())

                time.sleep(0.05)
    except KeyboardInterrupt:
        console.print(f"\n\n[bold yellow]Stopped by user[/bold yellow]")
        console.print(f"Total txs: {tx_count} | Alerts: {alert_count} ({100*alert_count/max(1,tx_count):.2f}%)")
    finally:
        try:
            w3.provider.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ws', default=os.environ.get('WEB3_WS'), help='Websocket URL for an Ethereum node (env WEB3_WS)')
    parser.add_argument('--threshold', type=float, default=0.94, help='Detection threshold (default: 0.935)')
    parser.add_argument('--lookback', type=int, default=50)
    parser.add_argument('--models-dir', default='models')
    args = parser.parse_args()

    if not args.ws:
        logger.error('Websocket URL is required via --ws or WEB3_WS env var')
        raise SystemExit(1)

    run_live_scoring(args.ws, threshold=args.threshold, lookback=args.lookback, models_dir=args.models_dir)
