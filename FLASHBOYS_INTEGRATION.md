# Flash Boys 2.0 Integration Guide

## Overview

We've integrated the **official Flash Boys 2.0 repository** as a submodule. This gives us access to the ACTUAL analysis tools used in the research paper that your algorithm is based on!

## What We Got

The flashboys2 repo contains:
1. **Real auction detection scripts** (Python)
2. **Transaction analysis tools** from the paper
3. **Modified go-ethereum** for monitoring
4. **Web visualization** (webapp/)
5. **Statistical analysis** scripts

## Key Files from Flash Boys 2.0

### 🔥 **MOST USEFUL FOR YOU:**

#### 1. **`read_csv.py`** - Auction Detection (GOLD!)
```python
# This is the ACTUAL auction detection algorithm from the paper!
# Location: flashboys2/read_csv.py

Key functions:
- get_individual_auctions(seen_list)  # Detects gas auctions from transaction stream
- should_filter_frontier()            # Filters out-of-order transactions
- normalize_auction_ids()             # Groups related auctions together
- postprocess_bid_list()              # Calculates price/time deltas

What it does:
- Reads raw transaction data
- Identifies "gas auctions" (3-second windows with multiple bids)
- Groups bids by sender/nonce
- Calculates bid dynamics (price increases, timing)
```

**WHY THIS MATTERS**: This is more sophisticated than simple heuristics! It uses:
- **Time-based clustering** (3-second windows)
- **Nonce tracking** (filters sync issues)
- **Repeated bidding detection** (identifies serious attackers)

#### 2. **`calculate_profit_from_logs.py`** - Profit Calculation
```python
# Calculates actual arbitrage profits from Solidity logs
# Location: flashboys2/calculate_profit_from_logs.py

Uses:
- exchanges.py (Uniswap, Kyber, Bancor parsers)
- Log parsing to track token swaps
- Graph analysis to calculate net profit

Result: eth_profit field for each transaction
```

#### 3. **`exchanges.py`** - DEX Log Parsers
```python
# Parsers for major DEXes
# Location: flashboys2/exchanges.py

Supported:
- Uniswap (ETH purchase/sale, token purchase)
- Kyber (trades)
- Bancor (conversions)

Functions:
- parse_uniswap_ethpurchase()
- parse_kyber()
- parse_bancor()
- get_trade_data_from_log_item()
```

#### 4. **`generate_graphs.py`** - Analysis & Visualization
```python
# Statistical analysis of auctions
# Location: flashboys2/generate_graphs.py

Generates:
- Revenue/profit/cost distributions
- Gas price dynamics
- Pairwise competition analysis
- Winner statistics
```

### 🔧 **Supporting Scripts:**

#### 5. **`scrape_gasauctions.py`** - Block Scraping
- Fetches top 10 transactions per block
- Gets receipts and logs
- Identifies high gas price transactions

#### 6. **`get_bq_*.py`** - BigQuery Integration
- `get_bq_txlist.py`: Fetch DEX transactions from BigQuery
- `get_bq_logs.py`: Get emitted logs from DEXes
- `get_bq_blocks.py`: Block-level data

#### 7. **`webapp/`** - Visualization Dashboard
- Flask web app
- Real-time auction visualization
- Gas price charts with Plotly

## How This Improves Your Project

### 🎯 **What You Can Integrate NOW:**

#### 1. **Better Auction Detection** (`read_csv.py`)
Your current approach labels individual transactions. The paper's approach:
- **Groups related bids** into auctions
- **Tracks bid evolution** (price increases over time)
- **Identifies repeated bidders** (serious MEV bots)

**Action**: Adapt `get_individual_auctions()` to work with your database.

#### 2. **Profit Calculation** (`calculate_profit_from_logs.py`)
Your labels have `victim_reward` and `attacker_reward` from game theory. Add:
- **Actual realized profit** from DEX logs
- **Token swap tracking**
- **Net profit after gas costs**

**Action**: Use `exchanges.py` parsers on your transaction logs.

#### 3. **Richer Features** for ML
Current features:
```python
# Your current features
['victim_gas_price', 'front_gas_price', 'back_gas_price',
 'victim_value', 'gas_price_ratio', 'tx_position_victim']
```

Add from Flash Boys:
```python
# Additional features from paper
['price_delta',           # How much attacker increased bid
 'price_percent_delta',   # % increase over previous bid
 'time_delta',            # Time between bids (ms)
 'self_price_delta',      # How much bot increased own bid
 'self_time_delta',       # Time between own bids
 'num_bids_in_auction',   # Total competing bids
 'auction_duration',      # How long auction lasted
 'profit_estimate']       # From DEX logs
```

#### 4. **Web Dashboard** (`webapp/`)
The paper has a Flask app that visualizes auctions in real-time:
- Gas price over time charts
- Transaction details
- Profit calculations
- Bidder identification

**Action**: Adapt for your data to visualize patterns.

## Integration Plan

### Phase 1: Enhanced Auction Detection (HIGHEST PRIORITY)

```python
# New file: auction_detector.py (based on flashboys2/read_csv.py)

from pathlib import Path
import sqlite3
import csv

class AuctionDetector:
    """
    Detects gas auctions using Flash Boys 2.0 methodology
    """
    
    def __init__(self, db_path="crypto_data.db"):
        self.db_path = db_path
    
    def get_transactions_time_ordered(self, limit=10000):
        """Get recent transactions ordered by time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT block_number, transaction_hash, from_address, 
                   gas_price, gas_limit, timestamp, nonce
            FROM transactions 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    
    def detect_auctions(self, time_window=3.0):
        """
        Detect gas auctions using 3-second time windows
        
        Args:
            time_window: Seconds between transactions to group as auction
        
        Returns:
            List of auctions, each auction is list of transaction dicts
        """
        txs = self.get_transactions_time_ordered()
        
        auctions = []
        current_auction = []
        
        for i in range(len(txs) - 1):
            prev_tx = txs[i]
            tx = txs[i+1]
            
            # Calculate time difference
            time_diff = abs(prev_tx[5] - tx[5]).total_seconds()
            
            if time_diff < time_window:
                # Part of current auction
                if len(current_auction) == 0:
                    current_auction = [prev_tx, tx]
                else:
                    current_auction.append(tx)
            else:
                # Auction ended
                if len(current_auction) >= 2:
                    auctions.append(current_auction)
                current_auction = []
        
        return auctions
    
    def calculate_auction_features(self, auction):
        """
        Calculate rich features for an auction
        """
        gas_prices = [tx[3] for tx in auction]
        timestamps = [tx[5] for tx in auction]
        
        return {
            'num_bids': len(auction),
            'price_range': max(gas_prices) - min(gas_prices),
            'duration': (max(timestamps) - min(timestamps)).total_seconds(),
            'avg_price': sum(gas_prices) / len(gas_prices),
            'num_unique_bidders': len(set(tx[2] for tx in auction))
        }
```

### Phase 2: Profit Calculation

```python
# New file: profit_calculator.py (based on flashboys2/calculate_profit_from_logs.py)

import sys
sys.path.append('flashboys2')
from exchanges import get_trade_data_from_log_item

class ProfitCalculator:
    """
    Calculate realized profits from DEX logs
    """
    
    def calculate_profit_from_transaction(self, tx_hash):
        """
        Parse logs to find DEX trades and calculate profit
        
        Uses exchanges.py from Flash Boys repo
        """
        # Get logs for transaction
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT log_index, address, data, topics
            FROM logs
            WHERE transaction_hash = ?
            ORDER BY log_index
        """, (tx_hash,))
        
        logs = cursor.fetchall()
        
        # Parse trades using Flash Boys parsers
        trades = []
        for log in logs:
            trade_data = get_trade_data_from_log_item(
                log[3],  # topics
                log[2],  # data
                log[1]   # address
            )
            if trade_data:
                trades.append(trade_data)
        
        # Calculate net profit
        return self._calculate_net_profit(trades)
```

### Phase 3: Enhanced Features

```python
# Update label_data.py to include Flash Boys features

def label_block_transactions_enhanced(self, block_number):
    """
    Enhanced labeling with Flash Boys methodology
    """
    # 1. Get all transactions
    txs = self.get_block_transactions(block_number)
    
    # 2. Detect auctions (not just sandwich patterns)
    detector = AuctionDetector(self.db_path)
    auctions = detector.detect_auctions_in_block(block_number)
    
    # 3. For each auction, calculate rich features
    labeled = []
    for auction in auctions:
        features = detector.calculate_auction_features(auction)
        
        # 4. Run game-theoretic labeling (your current Exec algorithm)
        game_theory_labels = self.execute_auction(auction)
        
        # 5. Calculate realized profits
        profit_calc = ProfitCalculator()
        realized_profits = profit_calc.calculate_for_auction(auction)
        
        # 6. Combine all features
        combined = {
            **features,              # Flash Boys features
            **game_theory_labels,    # Your Exec() algorithm labels
            **realized_profits       # Actual DEX profits
        }
        
        labeled.append(combined)
    
    return labeled
```

## Recommended Next Steps

### 1. **Short Term** (This Week):
- [x] Add flashboys2 as submodule
- [ ] Extract `read_csv.py` auction detection
- [ ] Run it on your existing data
- [ ] Compare results with your current labeling

### 2. **Medium Term** (Next 2 Weeks):
- [ ] Integrate DEX log parsing from `exchanges.py`
- [ ] Add profit calculation to your labels
- [ ] Enhance feature set with auction dynamics
- [ ] Re-train model with richer features

### 3. **Long Term** (Month):
- [ ] Set up webapp for visualization
- [ ] Implement pairwise analysis (bot competition)
- [ ] Add statistical reporting from `generate_graphs.py`
- [ ] Publish comparative study: Game Theory vs Realized Profits

## Key Insights from Flash Boys Code

### 1. **Auction Detection is Complex**
They use:
- 3-second time windows
- Nonce frontier tracking (filters out-of-order txs)
- Repeated bidding detection
- Auction normalization (groups related sub-auctions)

### 2. **Multiple Data Sources**
- Raw transaction data
- BigQuery for historical analysis
- Solidity logs for profit calculation
- Modified geth for latency measurement

### 3. **Focus on Dynamics**
Not just "is this front-running?" but:
- How do bots respond to each other?
- What's the bidding strategy?
- How much do they increase bids?
- What's the profit margin?

## Files to Copy/Adapt

**Priority 1** (Copy and adapt now):
- `flashboys2/read_csv.py` → `enhanced_auction_detection.py`
- `flashboys2/exchanges.py` → `dex_parsers.py`

**Priority 2** (Integrate next):
- `flashboys2/calculate_profit_from_logs.py` → Add to `label_data.py`
- `flashboys2/generate_graphs.py` → `analysis_tools.py`

**Priority 3** (Future enhancement):
- `flashboys2/webapp/` → Custom dashboard for your data

## Comparison: Your Approach vs Flash Boys

| Aspect | Your Implementation | Flash Boys Approach |
|--------|-------------------|-------------------|
| **Detection** | Game-theoretic auction (Exec) | Time-window clustering + nonce tracking |
| **Labels** | Victim/attacker rewards | Realized DEX profits |
| **Scope** | Per-transaction | Per-auction (grouped bids) |
| **Features** | Gas prices, positions | + Bid dynamics, timing, profits |
| **Validation** | Theoretical displacement | Actual profit calculation |

**BEST APPROACH**: Combine both!
- Use Flash Boys for **auction detection** and **feature extraction**
- Use your Exec() for **game-theoretic labels**
- Use DEX logs for **realized profit validation**

This gives you:
1. **Richer features** → Better ML model
2. **Multiple labels** → Cross-validation
3. **Real profit** → Business value estimation
4. **Bidding dynamics** → Strategy analysis

## Running Flash Boys Scripts

### Example: Detect Auctions

```bash
cd flashboys2

# 1. Prepare your data in their format
python write_csv.py  # Converts DB to CSV

# 2. Run auction detection
python read_csv.py   # Creates auctions.csv

# 3. Calculate profits
python calculate_profit_from_logs.py

# 4. Generate statistics
python generate_graphs.py
```

### Example: Use Their Parsers

```python
import sys
sys.path.append('flashboys2')

from exchanges import get_trade_data_from_log_item

# Parse a Uniswap log
topics = ['0x7f4091b46c33e918a0f3aa42307641d17bb67029427a5369e54b353984238705', ...]
data = '0x...'
address = '0x...'

trade_info = get_trade_data_from_log_item(topics, data, address)
print(trade_info)  # Token swaps, amounts, prices
```

## Conclusion

The Flash Boys 2.0 repository gives you:
1. **Proven algorithms** from published research
2. **Real-world validation** (used on actual Ethereum data)
3. **Complete pipeline** (collection → detection → analysis)
4. **Visualization tools** for understanding patterns

**Your unique contribution**: Combining their empirical approach with game-theoretic labeling creates a more powerful system than either alone!

Next: I'll create the integration scripts to merge their methodology with your Exec() algorithm.
