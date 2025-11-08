# MEV Detection: Data Collection → Analysis → ML Training

## Complete Workflow

### 1. Collect Data
```bash
python collect_data.py --batch-size 20 --delay 5
```
- Fetches Ethereum blocks continuously
- Stores in `data/crypto_data.db`
- Run for hours to collect enough data (1000+ blocks recommended)

### 2. Analyze & Label Data
```bash
python perfect_analysis.py
```
- Detects MEV patterns using Flash Boys 2.0 algorithm
- Advanced sandwich attack detection
- Gas price displacement analysis
- Outputs: `data/mev_analysis_results.csv` (ready for ML training)

### 3. Train Model
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load labeled data
df = pd.read_csv('data/mev_analysis_results.csv')

# Features
features = ['gas_price', 'value', 'gas_ratio', 'time_delta', 
            'position_in_block', 'same_contract_count']
X = df[features]
y = df['is_mev']  # or 'is_sandwich', 'is_arbitrage'

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

## Data Files

All data stored in `data/` folder:
- `crypto_data.db` - Raw blockchain transactions
- `mev_analysis_results.csv` - Labeled MEV patterns (for ML training)
- `labeled_training_data.csv` - Game-theoretic labels (alternative)

## Quick Commands

**Collect 1000 blocks:**
```bash
python collect_data.py --batch-size 50 --delay 2
# Wait ~1 hour
```

**Analyze everything:**
```bash
python perfect_analysis.py
```

**Check stats:**
```bash
sqlite3 data/crypto_data.db "SELECT COUNT(*) FROM transactions"
wc -l data/mev_analysis_results.csv
```

## What Each Script Does

- `collect_data.py` - Blockchain data collection
- `perfect_analysis.py` - MEV detection & labeling
- `label_data.py` - Alternative labeling (game-theoretic auction)

That's it. Simple and clean.
