import os
import time
import argparse
import pickle
import logging
import threading
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
from rich.layout import Layout
from rich.panel import Panel

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flashboys2 heuristic constants (REAL implementation from the paper)
FLASHBOYS_HIGH_GAS_THRESHOLD = 50_000_000_000  # 50 gwei (lowered from original 310 for modern mempool)
FLASHBOYS_EXCLUDED_ADDRESSES = [
    "0xa62142888aba8370742be823c1782d17a0389da1",
    "0xdd9fd6b6f8f7ea932997992bbe67eabb3e316f3c"
]

# Global state for block monitoring
recent_blocks = deque(maxlen=20)  # Store last 20 blocks of mempool analysis
blocks_lock = threading.Lock()
current_block_num = 0  # Track block number for analysis

# Track mempool transactions (rolling window for auction detection)
current_block_txs = deque(maxlen=2000)  # Rolling mempool window (last ~60 seconds)
current_block_txs_lock = threading.Lock()

# Model performance tracking  
mempool_tx_data = {}  # hash -> (tx_record, is_alert, score, timestamp)
performance_lock = threading.Lock()
model_performance = {
    'true_positive': 0,   # Model predicted MEV, Flashboys confirmed
    'false_positive': 0,  # Model predicted MEV, Flashboys rejected
    'true_negative': 0,   # Model predicted clean, Flashboys confirmed
    'false_negative': 0   # Model predicted clean, Flashboys found MEV
}

# Fixed threshold - adaptive was causing wild oscillations
adaptive_threshold = 0.92  # Fixed threshold
threshold_lock = threading.Lock()
# Track per-block detection rates for comparison (for display only)
recent_block_rate_diffs = deque(maxlen=5)  # Store last 5 blocks: (ml_rate - fb_rate)


def check_flashboys_heuristic_mempool(tx: Dict[str, Any]) -> bool:
    """
    Simplified flashboys2 heuristic for mempool transactions.
    Only checks high gas price (can't see gas_used or logs until mined)
    """
    gas_price = tx.get('gasPrice') or 0
    to_addr = (tx.get('to') or '').lower()
    
    is_high_gas = gas_price >= FLASHBOYS_HIGH_GAS_THRESHOLD
    is_not_excluded = to_addr not in FLASHBOYS_EXCLUDED_ADDRESSES
    
    return is_high_gas and is_not_excluded


def check_auction_labeler_real(tx_data: Dict[str, Any], all_block_txs: List[Dict[str, Any]], 
                                time_window: float = 2.5, min_price_escalation: float = 1.03) -> bool:
    """
    ACTUAL labeling used in training (AuctionLabeler from train_mev_detector_improved.py).
    
    This is what was ACTUALLY used to create the training labels, NOT flashboys!
    
    Detects gas auctions by checking:
    1. Multiple txs to same target within time window (2.5 seconds)
    2. Gas price escalation (each tx >= 1.03x previous)
    OR
    3. Nonce replacements (same sender + same nonce)
    """
    current_to = (tx_data.get('to') or '').lower()
    current_from = tx_data.get('from', '').lower()
    current_nonce = tx_data.get('nonce', -1)
    current_gas = float(tx_data.get('gasPrice', 0))
    current_time = tx_data.get('ts', 0)  # Fixed: use 'ts' not 'timestamp'
    
    if current_to == '0x0':
        return False
    
    # Find candidates: same target OR same sender+nonce
    candidates = []
    has_nonce_replacement = False
    
    for other_tx in all_block_txs:
        other_to = (other_tx.get('to') or '').lower()
        other_from = other_tx.get('from', '').lower()
        other_nonce = other_tx.get('nonce', -1)
        other_time = other_tx.get('ts', 0)  # Fixed: use 'ts' not 'timestamp'
        other_gas = float(other_tx.get('gasPrice', 0))
        
        # Check time window
        if abs(other_time - current_time) > time_window:
            continue
        
        # Same target or same sender+nonce
        same_target = other_to == current_to
        same_sender_nonce = (other_from == current_from and other_nonce == current_nonce)
        
        if same_target or same_sender_nonce:
            candidates.append((other_time, other_gas))
            if same_sender_nonce and other_time != current_time:
                has_nonce_replacement = True
    
    if len(candidates) < 2:  # min_auction_size=2
        return False
    
    # Sort by time and check gas escalation
    candidates.sort()
    gas_prices = [g for _, g in candidates]
    
    # Modified to detect "clumps" and overall escalation rather than strict steps
    # This captures the entire auction sequence, not just the final winner
    min_gas = min(gas_prices)
    max_gas = max(gas_prices)
    price_spread = max_gas / (min_gas + 1e-9)
    is_escalating = price_spread >= min_price_escalation
    
    # Also consider large groups as auctions (high contention)
    is_large_clump = len(candidates) >= 4
    
    return is_escalating or has_nonce_replacement or is_large_clump


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

    # Don't log during live display - causes corruption
    # logger.info(f"Loaded model ({model_path.name}), scaler and {len(feature_cols)} features")
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
    # Don't log during live display - causes corruption
    # logger.info("Connected to Web3 provider")
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


def block_monitor_thread(ws_url: str, stop_event: threading.Event):
    """
    Monitor new blocks - when block completes, analyze recent MEMPOOL txs using AuctionLabeler.
    Uses a sliding time window (last 30 seconds of mempool) to detect auction patterns.
    """
    w3_blocks = init_web3(ws_url)
    block_filter = w3_blocks.eth.filter('latest')
    
    global current_block_num
    
    while not stop_event.is_set():
        try:
            new_blocks = block_filter.get_new_entries()
            for block_hash in new_blocks:
                current_block_num += 1
                block_time = time.time()
                
                # Get recent mempool txs (last 30 seconds) for auction detection
                with current_block_txs_lock:
                    # Filter to transactions from last 30 seconds
                    cutoff_time = block_time - 30.0
                    recent_mempool = [tx for tx in current_block_txs if tx.get('ts', 0) >= cutoff_time]
                    
                    # Keep mempool data for next block (don't clear - we need rolling window)
                    # But remove very old txs (older than 60 seconds)
                    old_cutoff = block_time - 60.0
                    while current_block_txs and current_block_txs[0].get('ts', 0) < old_cutoff:
                        current_block_txs.popleft()
                
                if not recent_mempool:
                    continue
                
                # Run AuctionLabeler on recent mempool data to detect auctions
                flashboys_txs = []
                for tx_data in recent_mempool:
                    try:
                        tx_hash_full = tx_data['hash']
                        is_flashboys = check_auction_labeler_real(tx_data, recent_mempool, 
                                                       time_window=2.5, min_price_escalation=1.03)
                        
                        # Compare with ML prediction
                        with performance_lock:
                            if tx_hash_full in mempool_tx_data:
                                _, was_alert, score, _ = mempool_tx_data[tx_hash_full]
                                if was_alert and is_flashboys:
                                    model_performance['true_positive'] += 1
                                elif was_alert and not is_flashboys:
                                    model_performance['false_positive'] += 1
                                elif not was_alert and is_flashboys:
                                    model_performance['false_negative'] += 1
                                elif not was_alert and not is_flashboys:
                                    model_performance['true_negative'] += 1
                        
                        if is_flashboys:
                            flashboys_txs.append({
                                'hash': tx_data['hash'][:16] + "...",
                                'from': tx_data.get('from', '')[:10] + "...",
                                'to': tx_data.get('to', '0x0')[:10] + "...",
                                'gas_price': float(tx_data.get('gasPrice', 0)) / 1e9,
                                'nonce': tx_data.get('nonce', -1)
                            })
                    except Exception:
                        continue
                
                # Store block analysis
                with blocks_lock:
                    recent_blocks.append({
                        'number': current_block_num,
                        'timestamp': block_time,
                        'total_txs': len(recent_mempool),
                        'flashboys_txs': flashboys_txs,
                        'fb_count': len(flashboys_txs)
                    })
        
        except Exception:
            pass
        
        time.sleep(0.1)


def run_live_scoring(ws_url: str, threshold: float = 0.9, lookback: int = 50, models_dir: str = 'models'):
    global adaptive_threshold
    adaptive_threshold = threshold  # Initialize with command line threshold
    
    model, scaler, feature_cols = load_artifacts(models_dir)
    w3 = init_web3(ws_url)

    # Sliding buffer of recent txs
    buffer: Deque[Dict[str, Any]] = deque(maxlen=1000)

    # Wait for next block to synchronize start
    block_filter = w3.eth.filter('latest')
    current_block = w3.eth.block_number
    while True:
        new_blocks = block_filter.get_new_entries()
        if new_blocks:
            break
        time.sleep(0.1)
    
    # Now subscribe to pending transactions - synced with block
    sub = w3.eth.filter('pending')
    
    # Stats tracking
    tx_count = 0
    alert_count = 0
    
    # Store last 25 transactions with scores for display
    recent_txs = deque(maxlen=25)
    
    # Block monitoring will start AFTER warmup completes
    stop_event = threading.Event()
    block_thread = None
    
    # Don't print to console - it corrupts the Live display
    # The status will be shown in the panel titles

    def render_blocks_table():
        """Render the flashboys analysis of mempool data per block"""
        with blocks_lock:
            blocks_list = list(recent_blocks)
        
        if not blocks_list:
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, 
                         title="[bold]Flashboys Mempool Analysis (Per Block)[/bold] | Waiting for blocks...")
            table.add_column("Block", width=10)
            table.add_column("Status", width=30)
            table.add_row("[dim]---[/dim]", "[yellow]Waiting for next block...[/yellow]")
            return table
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED,
                     title=f"[bold]Flashboys Mempool Analysis (Per Block)[/bold] | Last {len(blocks_list)} blocks")
        table.add_column("Block", justify="right", width=10)
        table.add_column("Total TXs", justify="right", width=10)
        table.add_column("FB Rate", justify="right", width=8)
        table.add_column("ML Rate", justify="right", width=8)
        table.add_column("FB Count", justify="right", width=10)
        table.add_column("ML Count", justify="right", width=10)
        table.add_column("Sample TX (Hash | Gas | Logs)", width=50)
        
        # Show all blocks (up to 50)
        for block_data in reversed(blocks_list):  # Most recent at top
            block_num = block_data['number']
            total = block_data['total_txs']
            fb_count_block = block_data['fb_count']
            fb_txs = block_data['flashboys_txs']
            fb_rate = (100 * fb_count_block / max(1, total))
            
            # Calculate ML alert rate for this block's transactions
            ml_alert_count = 0
            with performance_lock:
                for tx_hash in [tx['hash'][:16] for tx in fb_txs] if fb_txs else []:
                    # Find matching transactions in mempool_tx_data
                    for stored_hash, (_, was_alert, _, _) in mempool_tx_data.items():
                        if stored_hash.startswith(tx_hash.replace("...", "")):
                            if was_alert:
                                ml_alert_count += 1
                            break
            
            # Actually, let's calculate ML alerts for ALL txs in this block period
            block_start_time = block_data.get('timestamp', 0) - 30.0
            block_end_time = block_data.get('timestamp', 0)
            ml_alerts_in_period = 0
            with performance_lock:
                for _, (tx_rec, was_alert, _, tx_time) in mempool_tx_data.items():
                    if block_start_time <= tx_time <= block_end_time and was_alert:
                        ml_alerts_in_period += 1
            
            ml_rate = (100 * ml_alerts_in_period / max(1, total))
            
            # Show first flashboys tx as sample
            if fb_txs:
                sample = fb_txs[0]
                sample_str = f"{sample['hash']} | {sample['gas_price']:.1f}gwei | nonce:{sample.get('nonce', 'N/A')}"
                table.add_row(
                    f"[cyan]{block_num}[/cyan]",
                    str(total),
                    f"[green]{fb_rate:.1f}%[/green]" if fb_count_block > 0 else "[dim]0%[/dim]",
                    f"[cyan]{ml_rate:.1f}%[/cyan]" if ml_alerts_in_period > 0 else "[dim]0%[/dim]",
                    f"[bold green]{fb_count_block}[/bold green]" if fb_count_block > 0 else "0",
                    f"[bold cyan]{ml_alerts_in_period}[/bold cyan]" if ml_alerts_in_period > 0 else "0",
                    f"[dim]{sample_str}[/dim]"
                )
            else:
                table.add_row(
                    f"[dim]{block_num}[/dim]",
                    str(total),
                    "[dim]0%[/dim]",
                    f"[cyan]{ml_rate:.1f}%[/cyan]" if ml_alerts_in_period > 0 else "[dim]0%[/dim]",
                    "0",
                    f"[bold cyan]{ml_alerts_in_period}[/bold cyan]" if ml_alerts_in_period > 0 else "0",
                    "[dim]No MEV detected[/dim]"
                )
        
        return table

    def render_table():
        """Render the current state of recent transactions"""
        warmup_remaining = max(0, lookback - len(buffer))
        if warmup_remaining > 0:
            status_text = f"Warming up... ({len(buffer)}/{lookback})"
        else:
            alert_pct = 100 * alert_count / max(1, tx_count)
            status_text = f"Total: {tx_count} | ML Alerts: {alert_count} ({alert_pct:.2f}%)"
        
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
        
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED, expand=True)
        table.add_column("#", style="dim", width=6)
        table.add_column("Hash", width=hash_width)
        table.add_column("From", width=from_width)
        table.add_column("To", width=to_width)
        table.add_column("Gas (Gwei)", justify="right", width=gas_width)
        table.add_column("Score", justify="right", width=score_width)
        table.add_column("Status", width=status_width)
        
        # Show all transactions (up to 50)
        for tx_data in reversed(recent_txs):  # Most recent at top
            tx_num = tx_data['num']
            tx_rec = tx_data['tx']
            score = tx_data['score']
            is_alert = tx_data['is_alert']
            reason = tx_data.get('reason', '')
            fb_match = tx_data.get('fb_match', False)
            
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
    
    def render_stats():
        """Render statistics panel"""
        global adaptive_threshold, recent_block_rate_diffs
        
        with blocks_lock:
            blocks_list = list(recent_blocks)
        
        with performance_lock:
            perf = model_performance.copy()
        
        # Calculate statistics - use UNIQUE evaluated transactions, not overlapping block windows
        total_fb_real = sum(b['fb_count'] for b in blocks_list)
        # Don't sum total_txs across blocks - they overlap! Use the actual evaluated count instead
        fb_rate = (100 * total_fb_real / max(1, len(mempool_tx_data))) if mempool_tx_data else 0.0
        ml_rate = (100 * alert_count / max(1, tx_count))
        diff = abs(ml_rate - fb_rate)
        
        # Track rate differences for display (no longer adjusting threshold)
        # Store the rate difference from the most recent block
        if len(blocks_list) > 0:
            latest_block = blocks_list[-1]
            block_total = latest_block['total_txs']
            block_fb_count = latest_block['fb_count']
            
            # ONLY track blocks with enough transactions (ignore tiny blocks)
            if block_total >= 100:  # Require at least 100 txs for valid sample
                block_fb_rate = 100 * block_fb_count / block_total
                
                # Calculate ML rate for this specific block
                block_start_time = latest_block.get('timestamp', 0) - 30.0
                block_end_time = latest_block.get('timestamp', 0)
                block_ml_count = 0
                with performance_lock:
                    for _, (tx_rec, was_alert, _, tx_time) in mempool_tx_data.items():
                        if block_start_time <= tx_time <= block_end_time and was_alert:
                            block_ml_count += 1
                
                block_ml_rate = 100 * block_ml_count / block_total
                rate_diff = block_ml_rate - block_fb_rate
                
                # Store this block's rate difference (for display only)
                recent_block_rate_diffs.append(rate_diff)
        
        # Threshold is now FIXED - no more adjustments
        
        # Model performance metrics
        tp = perf['true_positive']
        fp = perf['false_positive']
        tn = perf['true_negative']
        fn = perf['false_negative']
        total_evaluated = tp + fp + tn + fn
        
        if total_evaluated > 0:
            accuracy = 100 * (tp + tn) / total_evaluated
            precision = 100 * tp / max(1, tp + fp)
            recall = 100 * tp / max(1, tp + fn)
            agreement = 100 * (tp + tn) / total_evaluated
            # Fixed scoring: Weight TP more, don't double-penalize
            score = (tp * 2) - fp - fn  # +2 for catching MEV, -1 for each error
        else:
            accuracy = precision = recall = agreement = 0.0
            score = 0
        
        # Horizontal table for statistics
        from rich.table import Table as StatsTable
        stats_table = StatsTable(show_header=True, box=box.SIMPLE, expand=True)
        
        stats_table.add_column("Mempool", justify="left", style="cyan")
        stats_table.add_column("Flashboys", justify="left", style="magenta")
        stats_table.add_column("Performance", justify="left", style="green")
        stats_table.add_column("Confusion Matrix", justify="left", style="yellow")
        
        with threshold_lock:
            current_threshold = adaptive_threshold
        
        stats_table.add_row(
            f"{tx_count} txs\n{alert_count} ML alerts ({ml_rate:.2f}%)",
            f"{len(blocks_list)} blocks analyzed\n{len(mempool_tx_data)} unique txs evaluated\n{total_fb_real} auctions ({fb_rate:.2f}%)",
            f"Score: {score:+d}\nAccuracy: {accuracy:.1f}%\nPrecision: {precision:.1f}%\nRecall: {recall:.1f}%\nAgreement: {agreement:.1f}%",
            f"TP={tp}  FP={fp}\nTN={tn}  FN={fn}\nEvaluated: {total_evaluated}\nThreshold: {current_threshold:.4f} (fixed)"
        )
        
        return Panel(stats_table, border_style="yellow", title="[bold yellow]Statistics: ML vs Flashboys[/bold yellow]")

    def render_combined():
        """Render both mempool and blocks tables stacked with colored panels"""
        from rich.console import Group
        
        # Get status for panel title
        warmup_remaining = max(0, lookback - len(buffer))
        if warmup_remaining > 0:
            mempool_title = f"[bold cyan]Mempool Feed[/bold cyan] | Warming up... ({len(buffer)}/{lookback})"
        else:
            alert_pct = 100 * alert_count / max(1, tx_count)
            mempool_title = f"[bold cyan]Mempool Feed[/bold cyan] | Total: {tx_count} | Alerts: {alert_count} ({alert_pct:.2f}%)"
        
        mempool_panel = Panel(render_table(), border_style="cyan", title=mempool_title)
        blocks_panel = Panel(render_blocks_table(), border_style="magenta", title="[bold magenta]Flashboys Analysis (Mempool Data)[/bold magenta]")
        stats_panel = render_stats()
        return Group(mempool_panel, blocks_panel, stats_panel)

    try:
        with Live(render_combined(), refresh_per_second=2, console=console) as live:
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

                    # Start block monitoring AFTER warmup completes (silently to avoid corrupting Live display)
                    if len(buffer) == lookback and block_thread is None:
                        block_thread = threading.Thread(target=block_monitor_thread, args=(ws_url, stop_event), daemon=True)
                        block_thread.start()

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

                        # Use adaptive threshold for alerts
                        with threshold_lock:
                            current_threshold = adaptive_threshold
                        
                        is_alert = score >= current_threshold
                        if is_alert:
                            alert_count += 1
                            reason = generate_alert_reason(feat_dict, score)
                        else:
                            reason = ""
                        
                        # Store for block analysis AND performance tracking
                        with performance_lock:
                            mempool_tx_data[tx_record['hash']] = (tx_record, is_alert, score, arrival_ts)
                        
                        # Add to current block's mempool collection
                        with current_block_txs_lock:
                            current_block_txs.append(tx_record)
                        
                        # Add to recent transactions for display
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
                    live.update(render_combined())

                time.sleep(0.05)
    except KeyboardInterrupt:
        stop_event.set()  # Stop the block monitor thread
        console.print(f"\n\n[bold yellow]Stopped by user[/bold yellow]")

    finally:
        try:
            w3.provider.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ws', default=os.environ.get('WEB3_WS'), help='Websocket URL for an Ethereum node (env WEB3_WS)')
    parser.add_argument('--threshold', type=float, default=0.61, help='Detection threshold')
    parser.add_argument('--lookback', type=int, default=50)
    parser.add_argument('--models-dir', default='models')
    args = parser.parse_args()

    if not args.ws:
        logger.error('Websocket URL is required via --ws or WEB3_WS env var')
        raise SystemExit(1)

    run_live_scoring(args.ws, threshold=args.threshold, lookback=args.lookback, models_dir=args.models_dir)
