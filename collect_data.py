#!/usr/bin/env python3
"""
Continuous Ethereum Data Collection

Collects Ethereum blockchain data continuously until interrupted.
Designed to run for hours collecting historical and live data.
"""

import asyncio
import sys
import sqlite3
import logging
import signal
from pathlib import Path
from datetime import datetime
import time
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousDataCollector:
    """Continuously collects Ethereum blockchain data"""
    
    def __init__(self, db_path: str = "data/crypto_data.db"):
        self.db_path = db_path
        self.should_stop = False
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        self.blocks_collected = 0
        self.start_time = None
        # Recent block collection timestamps for rate estimation
        self.recent_block_times = deque(maxlen=120)  # keep up to last 2 minutes at ~1 block/sec

        # Dynamic scaling state
        self.current_workers = 8  # Start conservative
        self.max_workers_limit = 32  # Hard cap
        self.min_workers = 2
        self.current_batch_size = 10  # Dynamic batch size
        self.max_batch_size = 100  # Hard cap for batch size
        self.min_batch_size = 5
        self.recent_fetch_times = deque(maxlen=50)  # Track fetch performance
        self.recent_errors = deque(maxlen=20)  # Track error rate
        self.scaling_cooldown = 0  # Batches to wait before next adjustment
        self.last_scale_time = 0

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\nShutdown signal received. Finishing current block...")
        self.should_stop = True
    
    def get_latest_collected_block(self) -> int:
        """Get the highest block number already collected"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Prefer explicit checkpoint table (more robust) if present
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'
            """)
            if cursor.fetchone():
                cursor.execute("SELECT last_block FROM checkpoints ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row and row[0] is not None:
                    conn.close()
                    return int(row[0])

            # Fallback: derive from transactions table
            cursor.execute("SELECT MAX(block_number) FROM transactions")
            result = cursor.fetchone()[0]
            conn.close()
            return int(result) if result else 0
        except Exception:
            return 0

    def _write_checkpoint(self, block_number: int):
        """Atomically write the last processed block to a checkpoints table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_block INTEGER,
                    updated_at DATETIME
                )
            """)
            cursor.execute(
                "INSERT INTO checkpoints (last_block, updated_at) VALUES (?, ?)",
                (int(block_number), datetime.utcnow())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to write checkpoint for block {block_number}: {e}")
    
    def get_database_stats(self):
        """Get current database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(DISTINCT block_number) FROM transactions")
            blocks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transactions")
            txs = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM transactions")
            block_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'blocks': blocks or 0,
                'transactions': txs or 0,
                'block_range': block_range if block_range[0] else (0, 0)
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'blocks': 0, 'transactions': 0, 'block_range': (0, 0)}
    
    async def collect_continuous(self, batch_size: int = 10, delay_between_batches: float = 5.0, max_workers: int = 4):
        """
        Collect data continuously until stopped
        
        Args:
            batch_size: Number of blocks to collect per batch
            delay_between_batches: Seconds to wait between batches
            max_workers: Maximum concurrent block fetches (default: 4)
        """
        try:
            from web3 import Web3
            import aiohttp
            from concurrent.futures import ThreadPoolExecutor
            
            # Initialize dynamic batch size
            self.current_batch_size = batch_size
            self.max_batch_size = min(100, batch_size * 10)  # Allow 10x growth from starting size
            
            logger.info("="*70)
            logger.info("CONTINUOUS ETHEREUM DATA COLLECTION - DYNAMIC SCALING")
            logger.info("="*70)
            logger.info(f"Starting batch size: {batch_size} blocks (max: {self.max_batch_size})")
            logger.info(f"Starting workers: {self.current_workers} (max: {max_workers})")
            logger.info(f"Delay between batches: {delay_between_batches}s")
            logger.info("Dynamic scaling: ENABLED (workers + batch size + multi-RPC)")
            logger.info("Press Ctrl+C to stop gracefully")
            logger.info("="*70)
            
            # Multiple RPC endpoints for load distribution
            rpc_endpoints = [
                "https://eth.llamarpc.com",
                "https://rpc.ankr.com/eth",
                "https://ethereum.publicnode.com",
                "https://1rpc.io/eth",
                "https://eth.rpc.blxrbdn.com",
                "https://virginia.rpc.blxrbdn.com",
                "https://uk.rpc.blxrbdn.com",
                "https://singapore.rpc.blxrbdn.com"
            ]
            
            # Create Web3 instances for all RPCs
            logger.info(f"\nInitializing {len(rpc_endpoints)} RPC endpoints for load distribution...")
            w3_instances = []
            for rpc_url in rpc_endpoints:
                try:
                    w3_inst = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 60}))
                    if w3_inst.is_connected():
                        w3_instances.append(w3_inst)
                        logger.info(f"  Connected: {rpc_url}")
                except Exception:
                    pass
            
            if not w3_instances:
                logger.error("Failed to connect to any RPC endpoint")
                return
            
            logger.info(f"Successfully connected to {len(w3_instances)} RPC endpoints")
            w3 = w3_instances[0]  # Use first for chain queries
            self.rpc_index = 0  # Round-robin counter
            
            # Get starting point
            latest_block = w3.eth.block_number
            last_collected = self.get_latest_collected_block()
            
            stats = self.get_database_stats()
            if stats['blocks'] > 0:
                logger.info(f"\nCurrent database: {stats['blocks']} blocks, {stats['transactions']:,} transactions")
                logger.info(f"Block range: {stats['block_range'][0]:,} - {stats['block_range'][1]:,}")

            # Startup/resume log
            if last_collected and last_collected > 0:
                logger.info(f"Resuming collection from checkpoint. Last processed block: {last_collected:,}")
            else:
                logger.info("No checkpoint found. Starting fresh or using fallback start strategy.")
            
            # Determine collection strategy
            if last_collected == 0:
                # Start from recent blocks
                start_block = latest_block - 100
                logger.info(f"\nStarting from block {start_block:,} (100 blocks behind)")
            elif last_collected < latest_block - 100:
                # Resume historical collection
                start_block = last_collected + 1
                logger.info(f"\nResuming historical collection from block {start_block:,}")
            else:
                # Caught up, follow chain tip
                start_block = last_collected + 1
                logger.info(f"\nCaught up! Following chain tip from block {start_block:,}")
            
            logger.info(f"Latest block on chain: {latest_block:,}")
            logger.info(f"Blocks behind: {latest_block - start_block:,}")
            logger.info("")
            
            self.start_time = time.time()
            current_block = start_block
            
            # Set initial workers and max limit
            self.current_workers = min(4, max_workers)
            self.max_workers_limit = max_workers
            
            # Create thread pool for blocking Web3 calls (dynamic sizing)
            executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # Semaphore to limit concurrent fetches (will be recreated when scaling)
            fetch_semaphore = asyncio.Semaphore(self.current_workers)
            
            # Main collection loop
            while not self.should_stop:
                try:
                    # Check latest block periodically
                    if current_block % 50 == 0:
                        latest_block = await asyncio.get_event_loop().run_in_executor(
                            executor, w3.eth.__getattribute__, 'block_number'
                        )
                    
                    # If caught up, wait for new blocks
                    if current_block > latest_block:
                        logger.info(f"Caught up to block {latest_block:,}. Waiting for new blocks...")
                        await asyncio.sleep(12)  # Ethereum block time
                        latest_block = await asyncio.get_event_loop().run_in_executor(
                            executor, w3.eth.__getattribute__, 'block_number'
                        )
                        continue
                    
                    # Adjust workers dynamically (more frequently, every batch)
                    await self._adjust_workers_dynamically(fetch_semaphore)
                    
                    # Collect batch in parallel (use dynamic batch size)
                    batch_end = min(current_block + self.current_batch_size - 1, latest_block)
                    block_range = list(range(current_block, batch_end + 1))
                    logger.info(f"Collecting blocks {current_block:,} to {batch_end:,} (workers: {self.current_workers}, batch: {self.current_batch_size})...")
                    
                    # Fetch all blocks in parallel with timing and RPC rotation
                    batch_start_time = time.time()
                    fetch_tasks = []
                    for bn in block_range:
                        # Round-robin across available RPC instances
                        rpc_to_use = w3_instances[self.rpc_index % len(w3_instances)]
                        self.rpc_index += 1
                        fetch_tasks.append(
                            self._fetch_block_parallel(rpc_to_use, bn, executor, fetch_semaphore)
                        )
                    fetched_blocks = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    batch_fetch_time = time.time() - batch_start_time
                    
                    # Record batch performance
                    self.recent_fetch_times.append(batch_fetch_time / len(block_range) if block_range else 0)
                    
                    # Write blocks to DB in order and count errors
                    batch_errors = 0
                    for block_num, block_result in zip(block_range, fetched_blocks):
                        if self.should_stop:
                            break
                        
                        if isinstance(block_result, Exception):
                            logger.error(f"Error fetching block {block_num}: {block_result}")
                            batch_errors += 1
                            self.recent_errors.append(1)
                            continue
                        
                        if block_result:
                            await self._save_block_to_db_async(block_result, block_num)
                            self.blocks_collected += 1
                            self.recent_errors.append(0)
                        else:
                            batch_errors += 1
                            self.recent_errors.append(1)
                    
                    current_block = batch_end + 1
                    
                    # Progress update (use recent window for rate estimate)
                    elapsed = time.time() - self.start_time
                    # Compute recent blocks/min using recent_block_times
                    rate_bpm = 0.0
                    try:
                        if len(self.recent_block_times) >= 2:
                            dt = self.recent_block_times[-1] - self.recent_block_times[0]
                            if dt > 0:
                                rate_bpm = (len(self.recent_block_times) - 1) / dt * 60.0
                        elif elapsed > 0:
                            rate_bpm = (self.blocks_collected / elapsed) * 60.0
                    except Exception:
                        rate_bpm = 0.0

                    # Calculate how many blocks we are behind (if any) and ETA to catch up
                    try:
                        blocks_behind = max(0, latest_block - current_block + 1)
                    except Exception:
                        blocks_behind = 0

                    eta_str = 'unknown'
                    if rate_bpm > 0 and blocks_behind > 0:
                        eta_seconds = int((blocks_behind / rate_bpm) * 60)
                        h = eta_seconds // 3600
                        m = (eta_seconds % 3600) // 60
                        s = eta_seconds % 60
                        if h:
                            eta_str = f"{h}h{m}m{s}s"
                        else:
                            eta_str = f"{m}m{s}s"

                    # Show error rate in progress
                    recent_error_rate = sum(self.recent_errors) / len(self.recent_errors) if self.recent_errors else 0
                    logger.info(
                        f"Progress: {self.blocks_collected} collected | {rate_bpm:.1f} blocks/min | "
                        f"Workers: {self.current_workers} | Batch: {self.current_batch_size} | Errors: {recent_error_rate:.1%} | "
                        f"Behind: {blocks_behind:,} | ETA: {eta_str}"
                    )
                    
                    # Wait before next batch (rate limiting)
                    if not self.should_stop and current_block <= latest_block:
                        await asyncio.sleep(delay_between_batches)
                
                except Exception as e:
                    logger.error(f"Error in collection loop: {e}")
                    await asyncio.sleep(5)
            
            # Cleanup
            executor.shutdown(wait=True)
            
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.error("Install with: pip install web3 aiohttp")
            return
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            return
        finally:
            await self._print_final_stats()
    
    async def _adjust_workers_dynamically(self, semaphore):
        """Dynamically adjust worker count AND batch size based on performance metrics"""
        if len(self.recent_fetch_times) < 5 or len(self.recent_errors) < 5:
            return  # Not enough data yet (lowered threshold for faster response)
        
        # Calculate metrics
        avg_fetch_time = sum(self.recent_fetch_times) / len(self.recent_fetch_times)
        error_rate = sum(self.recent_errors) / len(self.recent_errors)
        
        old_workers = self.current_workers
        old_batch = self.current_batch_size
        
        # Decision logic for workers AND batch size (more aggressive)
        if error_rate > 0.20:  # >20% errors: scale down aggressively
            self.current_workers = max(self.min_workers, int(self.current_workers * 0.5))
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.6))
            if old_workers != self.current_workers or old_batch != self.current_batch_size:
                logger.warning(f"High error rate ({error_rate:.1%}), scaling DOWN: "
                             f"workers {old_workers}->{self.current_workers}, batch {old_batch}->{self.current_batch_size}")
        
        elif error_rate > 0.10:  # 10-20% errors: scale down
            self.current_workers = max(self.min_workers, self.current_workers - 1)
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 3)
            if old_workers != self.current_workers or old_batch != self.current_batch_size:
                logger.warning(f"Elevated errors ({error_rate:.1%}), reducing: "
                             f"workers {old_workers}->{self.current_workers}, batch {old_batch}->{self.current_batch_size}")
        
        elif error_rate < 0.03 and avg_fetch_time < 1.0:  # <3% errors and fast: scale up!
            if self.current_workers < self.max_workers_limit or self.current_batch_size < self.max_batch_size:
                # Moderate growth when things are going well
                self.current_workers = min(self.max_workers_limit, self.current_workers + 2)
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 4)
                if old_workers != self.current_workers or old_batch != self.current_batch_size:
                    logger.info(f"Performance good, scaling UP: "
                              f"workers {old_workers}->{self.current_workers}, batch {old_batch}->{self.current_batch_size}")
        
        elif error_rate < 0.08 and (self.current_workers < self.max_workers_limit or self.current_batch_size < self.max_batch_size):
            # Conservative growth
            self.current_workers = min(self.max_workers_limit, self.current_workers + 1)
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 2)
            if old_workers != self.current_workers or old_batch != self.current_batch_size:
                logger.info(f"Incrementing: "
                          f"workers {old_workers}->{self.current_workers}, batch {old_batch}->{self.current_batch_size}")
    
    async def _fetch_block_parallel(self, w3, block_num: int, executor, semaphore):
        """Fetch a single block with semaphore rate limiting"""
        async with semaphore:
            try:
                block = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    w3.eth.get_block,
                    block_num,
                    True  # Include transactions
                )
                return block
            except Exception as e:
                logger.error(f"Error fetching block {block_num}: {e}")
                return None
    
    async def _save_block_to_db_async(self, block, block_num: int):
        """Save a fetched block to database (async wrapper for sync DB ops)"""
        if not block:
            logger.warning(f"Block {block_num} not found")
            return
        
        try:
            # Call the standalone save function directly (it's defined later in this file)
            ok = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                self._save_block_sync,
                block
            )
            
            # Record timestamp for rate estimation
            try:
                self.recent_block_times.append(time.time())
            except Exception:
                pass
            
            logger.info(f"  Block {block_num}: {len(block.get('transactions', []))} transactions")
            
        except Exception as e:
            logger.error(f"Error saving block {block_num}: {e}")
    
    def _save_block_sync(self, block):
        """Synchronous DB save (called from executor)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if not exist
            cursor.execute("""
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
            
            cursor.execute("""
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
            
            # Insert block
            cursor.execute("""
                INSERT OR REPLACE INTO blocks 
                (block_number, block_hash, timestamp, miner, gas_used, gas_limit, base_fee, transaction_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(block['number']),
                block['hash'].hex() if block.get('hash') else '',
                datetime.fromtimestamp(block['timestamp']),
                block.get('miner', ''),
                block.get('gasUsed', 0),
                block.get('gasLimit', 0),
                block.get('baseFeePerGas', 0),
                len(block.get('transactions', []))
            ))
            
            # Insert transactions
            for tx in block.get('transactions', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO transactions
                    (block_number, transaction_hash, transaction_index, from_address, to_address,
                     value, gas_price, gas_used, gas_limit, timestamp, input_data, nonce, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(block['number']),
                    tx['hash'].hex(),
                    tx.get('transactionIndex', 0),
                    tx.get('from', ''),
                    tx.get('to', '') if tx.get('to') else '',
                    float(tx.get('value', 0)) / 1e18,
                    tx.get('gasPrice', 0),
                    tx.get('gas', 0),
                    tx.get('gas', 0),
                    datetime.fromtimestamp(block['timestamp']),
                    tx.get('input', ''),
                    tx.get('nonce', 0),
                    1
                ))
            
            conn.commit()
            
            # Write checkpoint
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        last_block INTEGER,
                        updated_at DATETIME
                    )
                """)
                cursor.execute("INSERT INTO checkpoints (last_block, updated_at) VALUES (?, datetime('now'))", 
                             (int(block['number']),))
                conn.commit()
            except Exception:
                pass
            
            conn.close()
            return True
        except Exception as e:
            logger.error(f"DB save error: {e}")
            return False
    
    async def _collect_single_block(self, w3, block_num: int, executor):
        """Collect a single block and its transactions (legacy method for compatibility)"""
        try:
            # Fetch block
            block = await asyncio.get_event_loop().run_in_executor(
                executor,
                w3.eth.get_block,
                block_num,
                True  # Include transactions
            )
            
            if not block:
                logger.warning(f"Block {block_num} not found")
                return
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if not exist
            cursor.execute("""
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
            
            cursor.execute("""
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
            
            # Insert block
            cursor.execute("""
                INSERT OR REPLACE INTO blocks 
                (block_number, block_hash, timestamp, miner, gas_used, gas_limit, base_fee, transaction_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block_num,
                block['hash'].hex() if block.get('hash') else '',
                datetime.fromtimestamp(block['timestamp']),
                block.get('miner', ''),
                block.get('gasUsed', 0),
                block.get('gasLimit', 0),
                block.get('baseFeePerGas', 0),
                len(block.get('transactions', []))
            ))
            
            # Insert transactions
            for tx in block.get('transactions', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO transactions
                    (block_number, transaction_hash, transaction_index, from_address, to_address,
                     value, gas_price, gas_used, gas_limit, timestamp, input_data, nonce, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    block_num,
                    tx['hash'].hex(),
                    tx.get('transactionIndex', 0),
                    tx.get('from', ''),
                    tx.get('to', '') if tx.get('to') else '',
                    float(tx.get('value', 0)) / 1e18,  # Convert to ETH
                    tx.get('gasPrice', 0),
                    tx.get('gas', 0),  # gas_used will be filled later with receipts
                    tx.get('gas', 0),  # gas_limit
                    datetime.fromtimestamp(block['timestamp']),
                    tx.get('input', ''),
                    tx.get('nonce', 0),
                    1  # Default status
                ))
            
            conn.commit()
            # Record timestamp for rate estimation
            try:
                self.recent_block_times.append(time.time())
            except Exception:
                pass

            # Update checkpoint after successful commit for this block
            try:
                self._write_checkpoint(block_num)
            except Exception:
                # non-fatal: we already committed transactions
                pass
            conn.close()
            
            logger.info(f"  Block {block_num}: {len(block.get('transactions', []))} transactions")
            
        except Exception as e:
            logger.error(f"Error collecting block {block_num}: {e}")
    
    async def _print_final_stats(self):
        """Print final statistics"""
        logger.info("\n" + "="*70)
        logger.info("COLLECTION SESSION COMPLETE")
        logger.info("="*70)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        hours = elapsed / 3600
        
        logger.info(f"Runtime: {hours:.2f} hours ({elapsed/60:.1f} minutes)")
        logger.info(f"Blocks collected: {self.blocks_collected}")
        
        if elapsed > 0:
            logger.info(f"Average rate: {self.blocks_collected/elapsed*60:.1f} blocks/min")
        
        stats = self.get_database_stats()
        logger.info(f"\nFinal database:")
        logger.info(f"  Total blocks: {stats['blocks']:,}")
        logger.info(f"  Total transactions: {stats['transactions']:,}")
        logger.info(f"  Block range: {stats['block_range'][0]:,} - {stats['block_range'][1]:,}")
        logger.info("="*70)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Continuous Ethereum Data Collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_data.py                    # Default: 10 blocks/batch, 5s delay
  python collect_data.py --batch-size 20    # Larger batches
  python collect_data.py --delay 10         # Slower rate (10s between batches)
        """
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        metavar='N',
        help='Number of blocks per batch (default: 10)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=5.0,
        metavar='SECONDS',
        help='Delay between batches in seconds (default: 5.0)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        metavar='N',
        help='Maximum concurrent block fetches (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import web3
        import aiohttp
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install web3 aiohttp")
        sys.exit(1)
    
    collector = ContinuousDataCollector()
    
    try:
        await collector.collect_continuous(
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
            max_workers=args.workers
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
