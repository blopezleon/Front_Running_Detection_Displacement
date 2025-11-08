#!/usr/bin/env python3

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
import aiohttp
from web3 import Web3
from collections import deque
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MempoolCollector:
    def __init__(self, db_path="data/mempool_realtime.db"):
        self.db_path = db_path
        self.tx_buffer = deque(maxlen=10000)
        self._init_db()
    
    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mempool_transactions (
                tx_hash TEXT PRIMARY KEY,
                from_address TEXT,
                to_address TEXT,
                gas_price INTEGER,
                gas_limit INTEGER,
                value TEXT,
                nonce INTEGER,
                timestamp INTEGER,
                data TEXT,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def store_transaction(self, tx):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO mempool_transactions 
                (tx_hash, from_address, to_address, gas_price, gas_limit, value, nonce, timestamp, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx['hash'],
                tx['from'],
                tx['to'],
                tx['gasPrice'],
                tx['gas'],
                str(tx['value']),
                tx['nonce'],
                int(datetime.now().timestamp()),
                tx.get('input', '')
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store transaction: {e}")
            return False
        finally:
            conn.close()
    
    async def connect_websocket(self, ws_url):
        logger.info(f"Connecting to: {ws_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newPendingTransactions"]
                }
                
                await ws.send_str(json.dumps(subscribe_msg))
                logger.info("Subscribed to pending transactions")
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        
                        if 'params' in data:
                            tx_hash = data['params']['result']
                            await self.fetch_transaction_details(tx_hash)
                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {msg.data}")
                        break
    
    async def fetch_transaction_details(self, tx_hash):
        pass
    
    def get_recent_transactions(self, limit=1000):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM mempool_transactions 
            ORDER BY received_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return rows

def main():
    collector = MempoolCollector()
    
    logger.info("Mempool collector initialized")
    logger.info("Note: Configure WebSocket URL in environment or code")
    logger.info(f"Data will be stored in: {collector.db_path}")
    
    logger.info("\nFor live collection, use:")
    logger.info("1. Infura WebSocket: wss://mainnet.infura.io/ws/v3/YOUR_KEY")
    logger.info("2. Alchemy WebSocket: wss://eth-mainnet.g.alchemy.com/v2/YOUR_KEY")
    logger.info("3. QuickNode WebSocket: wss://YOUR_ENDPOINT.quiknode.pro/YOUR_KEY")
    logger.info("\nOr download historical data from:")
    logger.info("https://mempool-dumpster.flashbots.net")

if __name__ == "__main__":
    main()
