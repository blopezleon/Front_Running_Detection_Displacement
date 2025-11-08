#!/usr/bin/env python3
"""db_inspect.py
Simple read-only inspector for the collector DB (data/crypto_data.db).
Prints file size, tables, transaction counts, block range, missing-block examples, and checkpoint summary.
This script is safe to run — it does NOT modify the DB.
"""
# Standard libraries
from pathlib import Path
import sqlite3
import sys
import argparse
import shutil
import time

# DB path and RPC endpoints
DB_PATH = Path('data/crypto_data.db')

RPC_ENDPOINTS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
    "https://ethereum.publicnode.com",
    "https://1rpc.io/eth",
]


def human(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def choose_rpc():
    # Return first reachable RPC endpoint
    from web3 import Web3
    for url in RPC_ENDPOINTS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 30}))
            if w3.is_connected():
                return w3
        except Exception:
            continue
    raise RuntimeError('No RPC endpoints available')


def open_db(path: Path):
    return sqlite3.connect(str(path))


def ensure_db_exists():
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH.resolve()}")
        sys.exit(1)

    print(f"Inspecting DB: {DB_PATH.resolve()}")
    size = DB_PATH.stat().st_size
    print(f"File size: {size} bytes ({human(size)})")

    conn = open_db(DB_PATH)
    cursor = conn.cursor()
    return conn, cursor
def inspect_db():
    conn, cursor = ensure_db_exists()

    # Tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print('\nTables found:', tables)

    if 'transactions' not in tables:
        print('\nNo transactions table found. Exiting.')
        conn.close()
        return []

    # Totals
    cursor.execute('SELECT COUNT(*) FROM transactions')
    tx_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT block_number) FROM transactions')
    blocks_count = cursor.fetchone()[0]
    print(f"\n{'='*70}")
    print(f"DATABASE SUMMARY")
    print(f"{'='*70}")
    print(f"Total transactions: {tx_count:,}")
    print(f"Distinct blocks: {blocks_count:,}")
    
    if blocks_count > 0:
        avg_tx_per_block = tx_count / blocks_count
        print(f"Average transactions per block: {avg_tx_per_block:.1f}")

    # Block range
    cursor.execute('SELECT MIN(block_number), MAX(block_number) FROM transactions')
    min_b, max_b = cursor.fetchone()
    print(f"\nBlock range: {min_b:,} -> {max_b:,}")

    # Time range analysis
    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM transactions')
    min_time, max_time = cursor.fetchone()
    if min_time and max_time:
        from datetime import datetime
        min_dt = datetime.fromisoformat(min_time) if isinstance(min_time, str) else min_time
        max_dt = datetime.fromisoformat(max_time) if isinstance(max_time, str) else max_time
        time_span = (max_dt - min_dt).total_seconds()
        hours = time_span / 3600
        print(f"Time span: {hours:.2f} hours ({time_span/60:.1f} minutes)")
        if hours > 0:
            blocks_per_hour = blocks_count / hours
            tx_per_hour = tx_count / hours
            print(f"Collection rate: {blocks_per_hour:.1f} blocks/hour, {tx_per_hour:,.0f} tx/hour")
    
    # Transaction statistics
    print(f"\n{'='*70}")
    print(f"TRANSACTION STATISTICS")
    print(f"{'='*70}")
    cursor.execute('SELECT MIN(value), MAX(value), AVG(value) FROM transactions WHERE value > 0')
    min_val, max_val, avg_val = cursor.fetchone()
    if min_val is not None:
        print(f"ETH value range: {min_val:.6f} -> {max_val:.6f} ETH")
        print(f"Average transaction value: {avg_val:.6f} ETH")
    
    cursor.execute('SELECT COUNT(*) FROM transactions WHERE value > 0')
    nonzero_tx = cursor.fetchone()[0]
    print(f"Transactions with ETH value: {nonzero_tx:,} ({nonzero_tx/tx_count*100:.1f}%)")
    
    cursor.execute('SELECT COUNT(DISTINCT from_address) FROM transactions')
    unique_senders = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT to_address) FROM transactions WHERE to_address != ""')
    unique_receivers = cursor.fetchone()[0]
    print(f"Unique addresses: {unique_senders:,} senders, {unique_receivers:,} receivers")
    
    # Block statistics
    if 'blocks' in tables:
        print(f"\n{'='*70}")
        print(f"BLOCK STATISTICS")
        print(f"{'='*70}")
        cursor.execute('SELECT AVG(gas_used), AVG(gas_limit), AVG(transaction_count) FROM blocks')
        avg_gas_used, avg_gas_limit, avg_tx_count = cursor.fetchone()
        if avg_gas_used:
            print(f"Average gas used: {avg_gas_used:,.0f}")
            print(f"Average gas limit: {avg_gas_limit:,.0f}")
            print(f"Average gas utilization: {avg_gas_used/avg_gas_limit*100:.1f}%")
        if avg_tx_count:
            print(f"Average transactions per block: {avg_tx_count:.1f}")
        
        cursor.execute('SELECT MIN(transaction_count), MAX(transaction_count) FROM blocks')
        min_tx_block, max_tx_block = cursor.fetchone()
        if min_tx_block is not None:
            print(f"Block transaction range: {min_tx_block} -> {max_tx_block} transactions")
    
    # Continuity check
    print(f"\n{'='*70}")
    print(f"DATA CONTINUITY")
    print(f"{'='*70}")
    missing = []
    if min_b is None:
        print('No blocks present.')
    else:
        span = max_b - min_b if max_b and min_b else 0
        print(f"Span: {span} blocks")

        # Build the set of recorded block_numbers from transactions and blocks table
        cursor.execute('SELECT DISTINCT block_number FROM transactions ORDER BY block_number')
        tx_blocks = [r[0] for r in cursor.fetchall()]
        cursor.execute("SELECT block_number FROM blocks")
        blk_rows = [r[0] for r in cursor.fetchall()] if 'blocks' in tables else []
        present_blocks = set(tx_blocks) | set(blk_rows)

        if span <= 20000:
            expected = set(range(min_b, max_b + 1))
            missing = sorted(list(expected - present_blocks))
            if missing:
                print(f"Missing blocks: {len(missing)} (examples: {missing[:10]})")
            else:
                print('Status: Fully sequential (no gaps)')
        else:
            print('Large span detected; sampling first/last 10k blocks...')
            missing_examples = []
            windows = [ (min_b, min(min_b+10000, max_b)), (max(max_b-10000, min_b), max_b) ]
            for wmin, wmax in windows:
                expected = set(range(wmin, wmax+1))
                present = set([b for b in present_blocks if wmin <= b <= wmax])
                missing_window = sorted(list(expected - present))
                if missing_window:
                    missing_examples.extend(missing_window[:20])
            if missing_examples:
                print(f'Missing blocks in sampled windows: {missing_examples[:10]}')
                missing = missing_examples
            else:
                print('Status: No gaps detected in sampled windows')
    
    print(f"{'='*70}")
    conn.close()
    return missing


def backup_db():
    ts = int(time.time())
    dst = DB_PATH.parent / f"crypto_data.db.bak.{ts}"
    print(f"Backing up DB to {dst}")
    shutil.copy2(DB_PATH, dst)
    return dst


def backfill_blocks(missing_blocks, max_backfill=100):
    if not missing_blocks:
        print('No missing blocks to backfill')
        return

    to_backfill = missing_blocks[:max_backfill]
    print(f"Preparing to backfill {len(to_backfill)} blocks (max {max_backfill})")

    # Backup DB first
    backup_db()

    # Connect to RPC
    try:
        w3 = choose_rpc()
        print(f"Using RPC: {w3.provider.endpoint_uri}")
    except Exception as e:
        print(f"Failed to connect to RPC: {e}")
        return

    conn = open_db(DB_PATH)
    cur = conn.cursor()

    # Ensure tables exist (same schema as collector)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS blocks (
        block_number INTEGER PRIMARY KEY,
        block_hash TEXT,
        timestamp DATETIME,
        miner TEXT,
        gas_used INTEGER,
        gas_limit INTEGER,
        base_fee INTEGER,
        transaction_count INTEGER
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_number INTEGER,
        transaction_hash TEXT UNIQUE,
        transaction_index INTEGER,
        from_address TEXT,
        to_address TEXT,
        value REAL,
        gas_price INTEGER,
        gas_used INTEGER,
        gas_limit INTEGER,
        timestamp DATETIME,
        input_data TEXT,
        nonce INTEGER,
        status INTEGER
    )
    """)

    from datetime import datetime

    for bn in to_backfill:
        try:
            print(f"Fetching block {bn} from RPC...")
            block = w3.eth.get_block(bn, full_transactions=True)
            if not block:
                print(f"  Block {bn} not found via RPC")
                continue

            # Insert/replace block
            cur.execute("""
            INSERT OR REPLACE INTO blocks
            (block_number, block_hash, timestamp, miner, gas_used, gas_limit, base_fee, transaction_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block['number'],
                block['hash'].hex() if block.get('hash') else '',
                datetime.utcfromtimestamp(block['timestamp']).isoformat(),
                block.get('miner', ''),
                block.get('gasUsed', 0),
                block.get('gasLimit', 0),
                block.get('baseFeePerGas', 0),
                len(block.get('transactions', []))
            ))

            # Insert transactions
            for tx in block.get('transactions', []):
                cur.execute("""
                INSERT OR REPLACE INTO transactions
                (block_number, transaction_hash, transaction_index, from_address, to_address,
                 value, gas_price, gas_used, gas_limit, timestamp, input_data, nonce, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    block['number'],
                    tx['hash'].hex(),
                    tx.get('transactionIndex', 0),
                    tx.get('from', ''),
                    tx.get('to', '') if tx.get('to') else '',
                    float(tx.get('value', 0)) / 1e18,
                    tx.get('gasPrice', 0),
                    tx.get('gas', 0),
                    tx.get('gas', 0),
                    datetime.utcfromtimestamp(block['timestamp']).isoformat(),
                    tx.get('input', ''),
                    tx.get('nonce', 0),
                    1
                ))

            conn.commit()
            print(f"  Backfilled block {bn}: {len(block.get('transactions', []))} txs")

            # Optionally write checkpoint table
            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        last_block INTEGER,
                        updated_at DATETIME
                    )
                """)
                cur.execute("INSERT INTO checkpoints (last_block, updated_at) VALUES (?, datetime('now'))", (int(bn),))
                conn.commit()
            except Exception:
                pass

        except Exception as e:
            print(f"Error backfilling block {bn}: {e}")
            conn.rollback()
            continue

    conn.close()
    print('Backfill complete.')


def main():
    parser = argparse.ArgumentParser(description='DB inspector and optional backfiller')
    parser.add_argument('--backfill', action='store_true', help='Attempt to backfill missing blocks (will backup DB)')
    parser.add_argument('--fill-gaps', action='store_true', help='Call collect_data backfill to fill missing blocks using collector logic')
    parser.add_argument('--max', type=int, default=100, help='Maximum missing blocks to backfill')
    args = parser.parse_args()

    missing = inspect_db()

    if args.backfill and missing:
        backfill_blocks(missing, max_backfill=args.max)

    # Optionally call the collector's backfill helper to fill gaps using the collector implementation
    if args.fill_gaps and missing:
        try:
            from collect_data import backfill_blocks_from_w3
        except Exception as e:
            print(f"Failed to import collect_data backfill helper: {e}")
        else:
            try:
                w3 = choose_rpc()
                print(f"Using RPC: {w3.provider.endpoint_uri}")
            except Exception as e:
                print(f"Failed to connect to RPC for fill-gaps: {e}")
            else:
                # Use a max slice of missing blocks
                to_fill = missing[:args.max]
                print(f"Invoking collector backfill for {len(to_fill)} blocks")
                backfill_blocks_from_w3(to_fill, w3, db_path=str(DB_PATH))



if __name__ == '__main__':
    main()
