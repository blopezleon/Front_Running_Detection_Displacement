
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from web3 import Web3
from datetime import datetime, timedelta
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TransactionData:
    """Data class for storing transaction information"""
    block_number: int
    transaction_hash: str
    transaction_index: int
    from_address: str
    to_address: str
    value: float
    gas_price: int
    gas_used: int
    gas_limit: int
    timestamp: datetime
    input_data: str
    nonce: int
    status: int

@dataclass
class MEVOpportunity:
    """Data class for storing MEV opportunity information"""
    block_number: int
    mev_type: str  # sandwich, arbitrage, liquidation, etc.
    profit_usd: float
    gas_cost_usd: float
    net_profit_usd: float
    victim_tx_hash: str
    frontrun_tx_hash: Optional[str]
    backrun_tx_hash: Optional[str]
    dex_name: str
    token_addresses: List[str]
    timestamp: datetime

class CryptoDataCollector:
    """Main class for collecting Ethereum data for front-running detection"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the data collector with configuration"""
        self.config = self._load_config(config_path)
        self.web3_clients = {}
        self.session = None
        self.db_path = "crypto_data.db"
        self.current_rpc = None
        self.rpc_index = 0
        self._setup_database()
        self._setup_web3_clients()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "ethereum_rpc": "https://eth.llamarpc.com",
            "fallback_rpcs": [
                "https://rpc.ankr.com/eth",
                "https://ethereum.publicnode.com",
                "https://eth.rpc.blxrbdn.com",
                "https://cloudflare-eth.com"
            ],
            "dune_api_key": "",
            "etherscan_api_key": "",
            "target_tokens": [
                "0xA0b86a33E6441E359e0DC2Db18db4ac7F08A2056",  # USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"   # WBTC
            ],
            "dex_contracts": {
                "uniswap_v2": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "uniswap_v3": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
                "sushiswap": "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
                "1inch": "0x1111111254EEB25477B68fb85Ed929f73A960582"
            },
            "block_range": 1000,
            "max_concurrent_requests": 2,
            "request_delay": 0.5,
            "retry_attempts": 3,
            "retry_delay": 2
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file at {config_path}")
        
        return default_config
    
    def _setup_database(self):
        """Setup SQLite database for storing collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Transactions table
        cursor.execute('''
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
        ''')
        
        # MEV opportunities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mev_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_number INTEGER,
                mev_type TEXT,
                profit_usd REAL,
                gas_cost_usd REAL,
                net_profit_usd REAL,
                victim_tx_hash TEXT,
                frontrun_tx_hash TEXT,
                backrun_tx_hash TEXT,
                dex_name TEXT,
                token_addresses TEXT,
                timestamp DATETIME
            )
        ''')
        
        # Block data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_number INTEGER UNIQUE,
                block_hash TEXT,
                timestamp DATETIME,
                miner TEXT,
                gas_used INTEGER,
                gas_limit INTEGER,
                base_fee REAL,
                transaction_count INTEGER
            )
        ''')
        
        # Token prices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT,
                price_usd REAL,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup completed")
    
    def _setup_web3_clients(self):
        """Setup Web3 client for Ethereum with fallback RPCs"""
        # Try primary RPC first
        rpcs_to_try = [self.config["ethereum_rpc"]]
        
        # Add fallback RPCs if available
        if "fallback_rpcs" in self.config:
            rpcs_to_try.extend(self.config["fallback_rpcs"])
        
        for rpc_url in rpcs_to_try:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 60}))
                
                if w3.is_connected():
                    self.web3_clients[1] = w3
                    self.current_rpc = rpc_url
                    logger.info(f"✅ Connected to Ethereum mainnet via {rpc_url}")
                    return
            except Exception as e:
                logger.warning(f"⚠️  Failed to connect to {rpc_url}: {e}")
                continue
        
        logger.error("❌ Failed to connect to any Ethereum RPC endpoint")
        raise ConnectionError("Could not connect to Ethereum network")
    
    @asynccontextmanager
    async def get_session(self):
        """Async context manager for HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        yield self.session
    
    async def get_latest_block_number(self, chain_id: int = 1) -> int:
        """Get the latest block number for a given chain"""
        if chain_id not in self.web3_clients:
            raise ValueError(f"Chain {chain_id} not supported")
        
        w3 = self.web3_clients[chain_id]
        # Run synchronous web3 call in executor
        loop = asyncio.get_event_loop()
        block_number = await loop.run_in_executor(None, lambda: w3.eth.block_number)
        logger.info(f"📍 Latest Ethereum block: {block_number}")
        return block_number
    
    async def get_block_data(self, block_number: int, chain_id: int = 1) -> Dict:
        """Get detailed block data including all transactions"""
        if chain_id not in self.web3_clients:
            raise ValueError(f"Chain {chain_id} not supported")
        
        w3 = self.web3_clients[chain_id]
        try:
            logger.info(f"🔄 Fetching block {block_number}...")
            # Run synchronous web3 call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            block = await loop.run_in_executor(None, lambda: w3.eth.get_block(block_number, full_transactions=True))
            
            logger.info(f"✅ Block {block_number} fetched: {len(block.transactions)} transactions")
            return {
                'block_number': block.number,
                'block_hash': block.hash.hex(),
                'timestamp': datetime.fromtimestamp(block.timestamp),
                'miner': block.miner,
                'gas_used': block.gasUsed,
                'gas_limit': block.gasLimit,
                'base_fee': getattr(block, 'baseFeePerGas', 0),
                'transactions': block.transactions,
                'transaction_count': len(block.transactions)
            }
        except Exception as e:
            logger.error(f"❌ Error fetching block {block_number}: {e}")
            return None
    
    def extract_transaction_data(self, tx, block_timestamp: datetime, chain_id: int = 1) -> TransactionData:
        """Extract transaction data into structured format"""
        return TransactionData(
            block_number=tx.blockNumber,
            transaction_hash=tx.hash.hex(),
            transaction_index=tx.transactionIndex,
            from_address=tx['from'],
            to_address=tx.to if tx.to else '',
            value=float(Web3.from_wei(tx.value, 'ether')),
            gas_price=tx.gasPrice,
            gas_used=0,  # Will be filled from receipt
            gas_limit=tx.gas,
            timestamp=block_timestamp,
            input_data=tx.input.hex() if tx.input else '',
            nonce=tx.nonce,
            status=1  # Will be updated from receipt
        )
    
    async def get_transaction_receipts(self, tx_hashes: List[str], chain_id: int = 1, pbar: Optional[tqdm] = None) -> Dict[str, Dict]:
        """Get transaction receipts in parallel batches with rate limiting and progress bar"""
        if chain_id not in self.web3_clients:
            raise ValueError(f"Chain {chain_id} not supported")
        
        w3 = self.web3_clients[chain_id]
        receipts = {}
        
        # Get settings
        batch_size = self.config.get('batch_size', 20)
        retry_attempts = self.config.get('retry_attempts', 2)
        retry_delay = self.config.get('retry_delay', 1)
        
        # Create progress bar for receipts if not provided
        if pbar is None:
            pbar = tqdm(total=len(tx_hashes), desc="  📥 Receipts", unit="tx", leave=False)
            own_pbar = True
        else:
            own_pbar = False
        
        # Process in batches for better performance
        for batch_start in range(0, len(tx_hashes), batch_size):
            batch_end = min(batch_start + batch_size, len(tx_hashes))
            batch = tx_hashes[batch_start:batch_end]
            
            # Process batch in parallel with semaphore
            semaphore = asyncio.Semaphore(batch_size)
            
            async def fetch_receipt(tx_hash: str):
                async with semaphore:
                    for attempt in range(retry_attempts):
                        try:
                            loop = asyncio.get_event_loop()
                            receipt = await loop.run_in_executor(
                                None,
                                lambda: w3.eth.get_transaction_receipt(tx_hash)
                            )
                            return tx_hash, {
                                'gas_used': receipt.gasUsed,
                                'status': receipt.status,
                                'logs': receipt.logs
                            }
                        except Exception as e:
                            if '429' in str(e) or 'Too Many Requests' in str(e):
                                if attempt < retry_attempts - 1:
                                    await asyncio.sleep(retry_delay * (attempt + 1))
                                else:
                                    return tx_hash, {'gas_used': 0, 'status': 0, 'logs': []}
                            else:
                                return tx_hash, {'gas_used': 0, 'status': 0, 'logs': []}
                    return tx_hash, {'gas_used': 0, 'status': 0, 'logs': []}
            
            # Fetch batch in parallel
            batch_results = await asyncio.gather(*[fetch_receipt(tx) for tx in batch])
            
            # Store results and update progress
            for tx_hash, receipt_data in batch_results:
                receipts[tx_hash] = receipt_data
                pbar.update(1)
            
            # Small delay between batches to avoid overwhelming the RPC
            if batch_end < len(tx_hashes):
                await asyncio.sleep(0.1)
        
        if own_pbar:
            pbar.close()
        
        success_count = sum(1 for r in receipts.values() if r['gas_used'] > 0)
        success_rate = (success_count / len(receipts) * 100) if receipts else 0
        logger.info(f"  ✅ {success_count}/{len(receipts)} receipts ({success_rate:.1f}% success)")
        return receipts
    
    async def detect_mev_opportunities(self, block_data: Dict, chain_id: int = 1) -> List[MEVOpportunity]:
        """Detect MEV opportunities in block data"""
        opportunities = []
        transactions = block_data['transactions']
        
        # Sort transactions by gas price (descending) to identify potential front-running
        sorted_txs = sorted(transactions, key=lambda tx: tx.gasPrice, reverse=True)
        
        # Look for sandwich attacks (high gas price transactions surrounding a victim)
        opportunities.extend(await self._detect_sandwich_attacks(sorted_txs, block_data, chain_id))
        
        # Look for arbitrage opportunities
        opportunities.extend(await self._detect_arbitrage(sorted_txs, block_data, chain_id))
        
        # Look for liquidations
        opportunities.extend(await self._detect_liquidations(sorted_txs, block_data, chain_id))
        
        return opportunities
    
    async def _detect_sandwich_attacks(self, transactions: List, block_data: Dict, chain_id: int) -> List[MEVOpportunity]:
        """Detect sandwich attacks in transaction list"""
        opportunities = []
        
        # Group transactions by target contract (DEX)
        dex_txs = {}
        for tx in transactions:
            if tx.to and tx.to.lower() in [addr.lower() for addr in self.config['dex_contracts'].values()]:
                if tx.to not in dex_txs:
                    dex_txs[tx.to] = []
                dex_txs[tx.to].append(tx)
        
        # Analyze each DEX for sandwich patterns
        for dex_addr, dex_transactions in dex_txs.items():
            if len(dex_transactions) < 3:
                continue
            
            # Sort by transaction index to maintain order
            dex_transactions.sort(key=lambda tx: tx.transactionIndex)
            
            # Look for sandwich pattern: high gas -> low gas -> high gas
            for i in range(len(dex_transactions) - 2):
                tx1, tx2, tx3 = dex_transactions[i:i+3]
                
                # Check if it's a potential sandwich (gas price pattern)
                if (tx1.gasPrice > tx2.gasPrice * 1.1 and 
                    tx3.gasPrice > tx2.gasPrice * 1.1 and
                    tx1['from'].lower() == tx3['from'].lower()):
                    
                    # Estimate profit (simplified)
                    estimated_profit = self._estimate_sandwich_profit(tx1, tx2, tx3)
                    
                    opportunity = MEVOpportunity(
                        block_number=block_data['block_number'],
                        mev_type='sandwich',
                        profit_usd=estimated_profit,
                        gas_cost_usd=self._calculate_gas_cost_usd(tx1, tx3, chain_id),
                        net_profit_usd=estimated_profit - self._calculate_gas_cost_usd(tx1, tx3, chain_id),
                        victim_tx_hash=tx2.hash.hex(),
                        frontrun_tx_hash=tx1.hash.hex(),
                        backrun_tx_hash=tx3.hash.hex(),
                        dex_name=self._get_dex_name(dex_addr),
                        token_addresses=self._extract_token_addresses([tx1, tx2, tx3]),
                        timestamp=block_data['timestamp']
                    )
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def _detect_arbitrage(self, transactions: List, block_data: Dict, chain_id: int) -> List[MEVOpportunity]:
        """Detect arbitrage opportunities"""
        opportunities = []
        
        # Look for transactions that interact with multiple DEXes in the same block
        multi_dex_txs = []
        for tx in transactions:
            if tx.to and len(tx.input) > 10:  # Has significant input data
                # Check if transaction interacts with multiple DEXes (simplified)
                dex_interactions = sum(1 for dex_addr in self.config['dex_contracts'].values() 
                                     if dex_addr.lower() in tx.input.hex().lower())
                if dex_interactions >= 2:
                    multi_dex_txs.append(tx)
        
        for tx in multi_dex_txs:
            estimated_profit = self._estimate_arbitrage_profit(tx)
            
            opportunity = MEVOpportunity(
                block_number=block_data['block_number'],
                mev_type='arbitrage',
                profit_usd=estimated_profit,
                gas_cost_usd=self._calculate_gas_cost_usd(tx, None, chain_id),
                net_profit_usd=estimated_profit - self._calculate_gas_cost_usd(tx, None, chain_id),
                victim_tx_hash='',
                frontrun_tx_hash=tx.hash.hex(),
                backrun_tx_hash='',
                dex_name='multiple',
                token_addresses=self._extract_token_addresses([tx]),
                timestamp=block_data['timestamp']
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _detect_liquidations(self, transactions: List, block_data: Dict, chain_id: int) -> List[MEVOpportunity]:
        """Detect liquidation opportunities"""
        opportunities = []
        
        # Look for transactions to lending protocols (simplified detection)
        lending_protocols = [
            '0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9',  # Aave
            '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b',  # Compound
        ]
        
        for tx in transactions:
            if (tx.to and tx.to.lower() in [addr.lower() for addr in lending_protocols] and
                'liquidate' in tx.input.hex().lower()):
                
                estimated_profit = self._estimate_liquidation_profit(tx)
                
                opportunity = MEVOpportunity(
                    block_number=block_data['block_number'],
                    mev_type='liquidation',
                    profit_usd=estimated_profit,
                    gas_cost_usd=self._calculate_gas_cost_usd(tx, None, chain_id),
                    net_profit_usd=estimated_profit - self._calculate_gas_cost_usd(tx, None, chain_id),
                    victim_tx_hash='',
                    frontrun_tx_hash=tx.hash.hex(),
                    backrun_tx_hash='',
                    dex_name=self._get_protocol_name(tx.to),
                    token_addresses=self._extract_token_addresses([tx]),
                    timestamp=block_data['timestamp']
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _estimate_sandwich_profit(self, front_tx, victim_tx, back_tx) -> float:
        """Estimate profit from sandwich attack (simplified)"""
        # This is a simplified estimation - in reality, you'd need to:
        # 1. Decode the transaction inputs to get exact amounts
        # 2. Calculate price impact on the AMM
        # 3. Estimate the profit from the price difference
        
        # For now, return a rough estimate based on transaction values
        front_value = float(Web3.from_wei(front_tx.value, 'ether'))
        victim_value = float(Web3.from_wei(victim_tx.value, 'ether'))
        back_value = float(Web3.from_wei(back_tx.value, 'ether'))
        
        # Rough estimate: 0.1-1% of victim transaction value
        return victim_value * 0.005  # 0.5% estimate
    
    def _estimate_arbitrage_profit(self, tx) -> float:
        """Estimate arbitrage profit (simplified)"""
        tx_value = float(Web3.from_wei(tx.value, 'ether'))
        # Rough estimate: 0.5-2% of transaction value
        return tx_value * 0.01  # 1% estimate
    
    def _estimate_liquidation_profit(self, tx) -> float:
        """Estimate liquidation profit (simplified)"""
        tx_value = float(Web3.from_wei(tx.value, 'ether'))
        # Liquidation bonuses are typically 5-10%
        return tx_value * 0.075  # 7.5% estimate
    
    def _calculate_gas_cost_usd(self, tx1, tx2=None, chain_id: int = 1) -> float:
        """Calculate gas cost in USD (simplified)"""
        # This would need real-time ETH/token price data
        eth_price_usd = 2000  # Placeholder
        
        gas_used = tx1.gas  # Would use actual gas used from receipt
        if tx2:
            gas_used += tx2.gas
        
        gas_cost_eth = Web3.from_wei(gas_used * tx1.gasPrice, 'ether')
        return float(gas_cost_eth) * eth_price_usd
    
    def _get_dex_name(self, address: str) -> str:
        """Get DEX name from contract address"""
        address_to_name = {v.lower(): k for k, v in self.config['dex_contracts'].items()}
        return address_to_name.get(address.lower(), 'unknown')
    
    def _get_protocol_name(self, address: str) -> str:
        """Get protocol name from contract address"""
        protocols = {
            '0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9': 'aave',
            '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b': 'compound'
        }
        return protocols.get(address.lower(), 'unknown')
    
    def _extract_token_addresses(self, transactions: List) -> List[str]:
        """Extract token addresses from transaction data (simplified)"""
        # In reality, you'd decode the transaction input to get exact token addresses
        # For now, return target tokens as placeholder
        return self.config['target_tokens'][:2]  # Return first 2 as example
    
    def get_collected_blocks(self) -> Set[int]:
        """Get set of block numbers already in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT block_number FROM transactions ORDER BY block_number')
            blocks = {row[0] for row in cursor.fetchall()}
            conn.close()
            return blocks
        except Exception as e:
            logger.warning(f"Could not read existing blocks: {e}")
            return set()
    
    def get_database_stats(self) -> Dict:
        """Get current database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM transactions')
            tx_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT block_number) FROM transactions')
            block_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT MIN(block_number), MAX(block_number) FROM transactions')
            min_block, max_block = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM mev_opportunities')
            mev_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_transactions': tx_count,
                'total_blocks': block_count,
                'block_range': (min_block, max_block) if min_block else (None, None),
                'mev_opportunities': mev_count
            }
        except Exception as e:
            logger.warning(f"Could not get database stats: {e}")
            return {
                'total_transactions': 0,
                'total_blocks': 0,
                'block_range': (None, None),
                'mev_opportunities': 0
            }
    
    async def save_transactions_to_db(self, transactions: List[TransactionData], chain_id: int = 1):
        """Save transaction data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for tx in transactions:
            cursor.execute('''
                INSERT OR REPLACE INTO transactions 
                (block_number, transaction_hash, transaction_index, from_address, to_address,
                 value, gas_price, gas_used, gas_limit, timestamp, input_data, nonce, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx.block_number, tx.transaction_hash, tx.transaction_index,
                tx.from_address, tx.to_address, tx.value, tx.gas_price,
                tx.gas_used, tx.gas_limit, tx.timestamp, tx.input_data,
                tx.nonce, tx.status
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(transactions)} transactions to database")
    
    async def save_mev_opportunities_to_db(self, opportunities: List[MEVOpportunity], chain_id: int = 1):
        """Save MEV opportunities to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for opp in opportunities:
            cursor.execute('''
                INSERT INTO mev_opportunities
                (block_number, mev_type, profit_usd, gas_cost_usd, net_profit_usd,
                 victim_tx_hash, frontrun_tx_hash, backrun_tx_hash, dex_name,
                 token_addresses, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp.block_number, opp.mev_type, opp.profit_usd, opp.gas_cost_usd,
                opp.net_profit_usd, opp.victim_tx_hash, opp.frontrun_tx_hash,
                opp.backrun_tx_hash, opp.dex_name, json.dumps(opp.token_addresses),
                opp.timestamp
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(opportunities)} MEV opportunities to database")
    
    async def collect_block_range_data(self, start_block: int, end_block: int, chain_id: int = 1, skip_existing: bool = True):
        """Collect data for a range of blocks with progress tracking"""
        
        # Get already collected blocks
        existing_blocks = self.get_collected_blocks() if skip_existing else set()
        
        # Filter out already collected blocks
        blocks_to_collect = [b for b in range(start_block, end_block + 1) if b not in existing_blocks]
        
        if not blocks_to_collect:
            logger.info(f"✅ All blocks {start_block} to {end_block} already collected!")
            return
        
        if existing_blocks:
            logger.info(f"📊 Found {len(existing_blocks)} blocks already in database")
            logger.info(f"📥 Collecting {len(blocks_to_collect)} new blocks ({start_block} to {end_block})")
        else:
            logger.info(f"📥 Collecting {len(blocks_to_collect)} blocks ({start_block} to {end_block})")
        
        semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
        
        # Create progress bar for blocks
        pbar = tqdm(total=len(blocks_to_collect), desc="🔄 Blocks", unit="block", position=0)
        
        async def process_block(block_number: int):
            async with semaphore:
                try:
                    # Get block data
                    block_data = await self.get_block_data(block_number, chain_id)
                    if not block_data:
                        pbar.update(1)
                        return
                    
                    # Extract transaction data
                    transactions = []
                    tx_hashes = []
                    
                    for tx in block_data['transactions']:
                        tx_data = self.extract_transaction_data(tx, block_data['timestamp'], chain_id)
                        transactions.append(tx_data)
                        tx_hashes.append(tx.hash.hex())
                    
                    # Get transaction receipts with progress
                    pbar.set_description(f"🔄 Block {block_number} ({len(tx_hashes)} txs)")
                    receipts = await self.get_transaction_receipts(tx_hashes, chain_id)
                    
                    # Update transaction data with receipt information
                    for tx_data in transactions:
                        if tx_data.transaction_hash in receipts:
                            receipt = receipts[tx_data.transaction_hash]
                            tx_data.gas_used = receipt['gas_used']
                            tx_data.status = receipt['status']
                    
                    # Detect MEV opportunities
                    mev_opportunities = await self.detect_mev_opportunities(block_data, chain_id)
                    
                    # Save to database
                    await self.save_transactions_to_db(transactions, chain_id)
                    if mev_opportunities:
                        await self.save_mev_opportunities_to_db(mev_opportunities, chain_id)
                    
                    # Update progress
                    pbar.set_postfix({'txs': len(transactions), 'MEV': len(mev_opportunities)})
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing block {block_number}: {e}")
                    pbar.update(1)
        
        # Process blocks concurrently
        tasks = [process_block(block_num) for block_num in blocks_to_collect]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        pbar.close()
        logger.info(f"✅ Collection complete! Processed {len(blocks_to_collect)} blocks")
    
    async def collect_latest_data(self, num_blocks: int = 10, chain_id: int = 1):
        """Collect data from the latest blocks"""
        latest_block = await self.get_latest_block_number(chain_id)
        start_block = latest_block - num_blocks + 1
        
        await self.collect_block_range_data(start_block, latest_block, chain_id)
    
    def export_data_for_training(self, output_dir: str = "training_data") -> Dict[str, str]:
        """Export collected data in formats suitable for RAG model training"""
        Path(output_dir).mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Export transactions
        tx_df = pd.read_sql_query('''
            SELECT * FROM transactions 
            ORDER BY block_number, transaction_index
        ''', conn)
        
        # Export MEV opportunities
        mev_df = pd.read_sql_query('''
            SELECT * FROM mev_opportunities 
            ORDER BY block_number, profit_usd DESC
        ''', conn)
        
        # Export blocks
        blocks_df = pd.read_sql_query('''
            SELECT * FROM blocks 
            ORDER BY block_number
        ''', conn)
        
        conn.close()
        
        # Save as CSV and Parquet for efficient loading
        file_paths = {}
        
        for name, df in [("transactions", tx_df), ("mev_opportunities", mev_df), ("blocks", blocks_df)]:
            csv_path = f"{output_dir}/{name}.csv"
            parquet_path = f"{output_dir}/{name}.parquet"
            
            df.to_csv(csv_path, index=False)
            df.to_parquet(parquet_path, index=False)
            
            file_paths[f"{name}_csv"] = csv_path
            file_paths[f"{name}_parquet"] = parquet_path
        
        # Create feature engineering dataset for ML
        if not tx_df.empty and not mev_df.empty:
            ml_features = self._create_ml_features(tx_df, mev_df)
            ml_csv_path = f"{output_dir}/ml_features.csv"
            ml_parquet_path = f"{output_dir}/ml_features.parquet"
            
            ml_features.to_csv(ml_csv_path, index=False)
            ml_features.to_parquet(ml_parquet_path, index=False)
            
            file_paths["ml_features_csv"] = ml_csv_path
            file_paths["ml_features_parquet"] = ml_parquet_path
        
        logger.info(f"Exported data to {output_dir}/")
        return file_paths
    
    def _create_ml_features(self, tx_df: pd.DataFrame, mev_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning model"""
        # Group transactions by block
        block_features = tx_df.groupby('block_number').agg({
            'gas_price': ['mean', 'std', 'min', 'max'],
            'value': ['sum', 'mean', 'std'],
            'gas_used': ['sum', 'mean'],
            'transaction_hash': 'count'
        }).reset_index()
        
        # Flatten column names
        block_features.columns = ['block_number', 'gas_price_mean', 'gas_price_std', 
                                'gas_price_min', 'gas_price_max', 'total_value', 
                                'avg_value', 'value_std', 'total_gas_used', 
                                'avg_gas_used', 'tx_count']
        
        # Add MEV labels
        mev_blocks = mev_df.groupby('block_number').agg({
            'profit_usd': 'sum',
            'mev_type': lambda x: ','.join(x.unique())
        }).reset_index()
        
        # Merge features with MEV data
        ml_features = block_features.merge(mev_blocks, on='block_number', how='left')
        ml_features['has_mev'] = ml_features['profit_usd'].notna().astype(int)
        ml_features['profit_usd'] = ml_features['profit_usd'].fillna(0)
        ml_features['mev_type'] = ml_features['mev_type'].fillna('none')
        
        return ml_features
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()


async def main():
    """Main function to demonstrate usage"""
    collector = CryptoDataCollector()
    
    try:
        # Collect data from latest 5 blocks on Ethereum
        logger.info("Starting Ethereum data collection...")
        await collector.collect_latest_data(num_blocks=5, chain_id=1)
        
        # Export data for training
        file_paths = collector.export_data_for_training()
        logger.info("Ethereum data collection completed!")
        logger.info(f"Exported files: {list(file_paths.keys())}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())

