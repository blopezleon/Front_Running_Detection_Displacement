# MEV Detection: Data Collection → Analysis → ML Training

## Complete Workflow (Official Flash Boys 2.0 Implementation)

### 1. Collect Data
```bash
python collect_data.py --batch-size 20 --delay 5
```
- Fetches Ethereum blocks continuously
- Stores in `data/crypto_data.db`
- Run for hours to collect enough data (1000+ blocks recommended)

### 2. Analyze with Flash Boys Algorithm
```bash
python flashboys_analysis.py
```
- **Uses OFFICIAL Flash Boys 2.0 algorithm from the paper authors**
- 3-second auction window detection
- Gas price competition analysis
- Outputs: `data/flashboys_analysis.csv` (ready for ML training)

### 3. Train Model
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load labeled data
df = pd.read_csv('data/flashboys_analysis.csv')

# Features
features = ['num_bids', 'num_bidders', 'max_gas_price', 'gas_price_range',
            'price_escalation_ratio', 'duration_seconds', 'blocks_spanned',
            'tx_position_in_auction', 'tx_gas_price']
X = df[features]
y = df['is_mev_auction']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

## Data Files

All data stored in `data/` folder:
- `crypto_data.db` - Raw blockchain transactions
- `flashboys_analysis.csv` - Flash Boys labeled data (ML ready) ⭐

## Quick Commands

**Collect data:**
```bash
python collect_data.py
```

**Analyze with Flash Boys algorithm:**
```bash
python flashboys_analysis.py
```

**Check stats:**
```bash
sqlite3 data/crypto_data.db "SELECT COUNT(*) FROM transactions"
wc -l data/flashboys_analysis.csv
```

## What Each Script Does

- `collect_data.py` - Blockchain data collection
- `flashboys_analysis.py` - **Official Flash Boys 2.0 auction detection** ⭐
- `label_data.py` - Alternative method (optional, not needed)

## About Flash Boys 2.0

The analysis uses the ACTUAL algorithm from the paper:
- Paper: "Flash Boys 2.0: Frontrunning, Transaction Reordering, and Consensus Instability in Decentralized Exchanges"
- Authors: Daian et al.
- Implementation: Based on `flashboys2/read_csv.py` from original researchers
- Method: 3-second auction windows, multi-bidder gas price competition

This is the professional, peer-reviewed implementation, not a guess.
