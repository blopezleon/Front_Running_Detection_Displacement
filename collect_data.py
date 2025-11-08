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
            cursor.execute("SELECT MAX(block_number) FROM transactions")
            result = cursor.fetchone()[0]
            conn.close()
            return result if result else 0
        except Exception:
            return 0
    
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
    
    async def collect_continuous(self, batch_size: int = 10, delay_between_batches: float = 5.0):
        """
        Collect data continuously until stopped
        
        Args:
            batch_size: Number of blocks to collect per batch
            delay_between_batches: Seconds to wait between batches
        """
        try:
            from web3 import Web3
            import aiohttp
            from concurrent.futures import ThreadPoolExecutor
            
            logger.info("="*70)
            logger.info("CONTINUOUS ETHEREUM DATA COLLECTION")
            logger.info("="*70)
            logger.info(f"Batch size: {batch_size} blocks")
            logger.info(f"Delay between batches: {delay_between_batches}s")
            logger.info("Press Ctrl+C to stop gracefully")
            logger.info("="*70)
            
            # Connect to Ethereum RPC
            rpc_url = "https://eth.llamarpc.com"
            logger.info(f"\nConnecting to RPC: {rpc_url}")
            
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 60}))
            
            if not w3.is_connected():
                logger.error("Failed to connect to Ethereum RPC")
                return
            
            logger.info("Connected to Ethereum RPC successfully")
            
            # Get starting point
            latest_block = w3.eth.block_number
            last_collected = self.get_latest_collected_block()
            
            stats = self.get_database_stats()
            if stats['blocks'] > 0:
                logger.info(f"\nCurrent database: {stats['blocks']} blocks, {stats['transactions']:,} transactions")
                logger.info(f"Block range: {stats['block_range'][0]:,} - {stats['block_range'][1]:,}")
            
            # Determine collection strategy
            if last_collected == 0:
                # Start from recent blocks
                start_block = latest_block - 100
                logger.info(f"\nStarting from block {start_block:,} (100 blocks behind)")
            elif last_collected < latest_block - 100:
                # Resume historical collection
                start_block = last_collected + 1
                logger.info(f"\nResuming from block {start_block:,}")
            else:
                # Caught up, follow chain tip
                start_block = last_collected + 1
                logger.info(f"\nCaught up! Following chain tip from block {start_block:,}")
            
            logger.info(f"Latest block on chain: {latest_block:,}")
            logger.info(f"Blocks behind: {latest_block - start_block:,}")
            logger.info("")
            
            self.start_time = time.time()
            current_block = start_block
            
            # Create thread pool for blocking Web3 calls
            executor = ThreadPoolExecutor(max_workers=4)
            
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
                    
                    # Collect batch
                    batch_end = min(current_block + batch_size - 1, latest_block)
                    logger.info(f"Collecting blocks {current_block:,} to {batch_end:,}...")
                    
                    for block_num in range(current_block, batch_end + 1):
                        if self.should_stop:
                            break
                        
                        await self._collect_single_block(w3, block_num, executor)
                        self.blocks_collected += 1
                    
                    current_block = batch_end + 1
                    
                    # Progress update
                    elapsed = time.time() - self.start_time
                    blocks_per_min = (self.blocks_collected / elapsed) * 60 if elapsed > 0 else 0
                    
                    logger.info(f"Progress: {self.blocks_collected} blocks collected "
                               f"({blocks_per_min:.1f} blocks/min)")
                    
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
    
    async def _collect_single_block(self, w3, block_num: int, executor):
        """Collect a single block and its transactions"""
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
            delay_between_batches=args.delay
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
