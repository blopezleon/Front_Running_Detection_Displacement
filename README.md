# Real-Time MEV Front-Running Detection

Professional ML model to detect MEV front-running auctions IN ACTION using mempool data.

## Key Innovation

Unlike FlashBoys (post-analysis), this model predicts MEV auctions BEFORE they complete, providing 1-5 second lead time for intervention.

## Architecture

1. **Data Collection**: Historical mempool data from Flashbots Mempool Dumpster
2. **Labeling**: FlashBoys algorithm for ground truth (3-second auction windows)
3. **Features**: 15+ temporal features from rolling windows (gas volatility, momentum, density)
4. **Model**: XGBoost classifier optimized for early detection
5. **Inference**: Real-time detector with transaction buffer for live deployment

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Historical Mempool Data

```bash
python download_mempool_historical.py 7
```

Downloads 7 days of mempool data (~5-7 GB total).

### Train Model

Open and run the Jupyter notebook:

```bash
jupyter notebook realtime_mev_detection.ipynb
```

The notebook will:
1. Load parquet files from `data/mempool/`
2. Apply FlashBoys labeling to detect gas auctions
3. Extract temporal features from mempool state
4. Train XGBoost classifier
5. Evaluate prediction lead time
6. Save model to `models/`

## Files

- `realtime_mev_detection.ipynb` - Main training notebook
- `download_mempool_historical.py` - Download Flashbots data
- `collect_mempool_data.py` - Real-time collection (requires WebSocket)
- `requirements.txt` - Python dependencies
- `mempool-dumpster/` - Flashbots submodule (data tools)

## Data Sources

**Historical**: https://mempool-dumpster.flashbots.net
- Daily parquet files with mempool transactions
- Includes timestamp, gas price, sender, receiver, nonce
- ~1-2M unique transactions per day

**Real-time** (configure WebSocket in collect_mempool_data.py):
- Infura: `wss://mainnet.infura.io/ws/v3/YOUR_KEY`
- Alchemy: `wss://eth-mainnet.g.alchemy.com/v2/YOUR_KEY`
- QuickNode: `wss://YOUR_ENDPOINT.quiknode.pro/YOUR_KEY`

## Model Performance

Expected metrics:
- ROC AUC: 0.85-0.95
- Prediction lead time: 1-5 seconds before auction completion
- Early detection: 60-80% of auctions detected before FlashBoys
- Precision: 70-85% (adjustable via threshold)

## Deployment

The trained model can be deployed for:
1. **Alert Systems**: Warn traders of incoming MEV activity
2. **Protection**: Insert counter-transactions before exploitation
3. **Monitoring**: Track MEV patterns in real-time
4. **Research**: Study auction precursors and dynamics

## References

FlashBoys 2.0 Paper: https://arxiv.org/pdf/1904.05234
