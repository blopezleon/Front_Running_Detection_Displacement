# 🚀 GETTING STARTED - No API Keys Required!

This guide will help you collect Ethereum front-running data using **completely free** public sources.

## ✅ What You Need

- Python 3.8+
- Internet connection
- About 5-10 minutes

**NO API KEYS NEEDED!** 🎉

---

## 📦 Step 1: Install Dependencies

```bash
pip install web3 pandas aiohttp
```

Or install everything:
```bash
pip install -r requirements.txt
```

---

## 🎯 Step 2: Run the Quick Start Script

### **Option A: Interactive Menu (Recommended)**

```bash
python quick_start.py
```

This will show you a menu with options:
- Collect 10 blocks (~2-3 minutes)
- Collect 50 blocks (~10-15 minutes)  
- Collect 100 blocks (~20-30 minutes)
- Analyze collected data
- Run full demo

### **Option B: Direct Data Collection**

```bash
python get_data.py
```

This collects 5 blocks by default using the free public Ethereum RPC.

---

## 📊 Step 3: Analyze the Data

```bash
python analyze_data.py
```

This will:
- Generate statistics about transactions
- Detect MEV opportunities
- Identify front-running patterns
- Create an HTML report

---

## 📈 What Gets Collected

### **From Free Public RPC (eth.llamarpc.com):**
- ✅ Block data
- ✅ Transaction details (gas prices, values, addresses)
- ✅ Gas usage patterns
- ✅ Transaction ordering

### **Detected Automatically:**
- 🎯 Sandwich attacks
- 💰 Arbitrage opportunities
- 📉 Liquidations
- ⚡ Front-running patterns

---

## 💾 Where Data is Stored

All data is saved to:
```
crypto_data.db          # SQLite database with all transactions
training_data/          # Exported CSV/Parquet files
ethereum_analysis_report.html  # Generated analysis report
```

---

## 🎓 Example Workflow

### **Quick Demo (5 minutes):**
```bash
# 1. Collect 10 blocks
python quick_start.py
# Choose option 1

# 2. View stats
# Choose option 5

# 3. Analyze
# Choose option 4

# 4. Open the HTML report
open ethereum_analysis_report.html
```

### **Standard Analysis (15 minutes):**
```bash
# 1. Collect 50 blocks
python quick_start.py
# Choose option 2

# 2. Analyze
# Choose option 4
```

### **Extended Research (30 minutes):**
```bash
# Collect 100 blocks
python quick_start.py
# Choose option 3
```

---

## 📊 Expected Results

### **10 Blocks (~1,500-2,000 transactions):**
- Database size: ~1-2 MB
- Processing time: 2-3 minutes
- Good for: Testing, quick demo

### **50 Blocks (~7,500-10,000 transactions):**
- Database size: ~5-10 MB
- Processing time: 10-15 minutes
- Good for: Initial analysis, pattern detection

### **100 Blocks (~15,000-20,000 transactions):**
- Database size: ~15-20 MB
- Processing time: 20-30 minutes
- Good for: Comprehensive analysis, model training

---

## 🔧 Customization

Edit `config.json` to change settings:

```json
{
  "ethereum_rpc": "https://eth.llamarpc.com",  // Free public RPC
  "max_concurrent_requests": 10,               // Speed vs rate limits
  "block_range": 1000                          // Max blocks per run
}
```

### **Alternative Free RPCs:**
- `https://eth.llamarpc.com` (default)
- `https://rpc.ankr.com/eth`
- `https://ethereum.publicnode.com`
- `https://eth.rpc.blxrbdn.com`

---

## 🎯 What You'll Learn

After running this system, you'll see:
1. **Gas Price Patterns**: How transactions compete for priority
2. **MEV Activity**: Real-world sandwich attacks and arbitrage
3. **Front-Running**: Transactions ordered suspiciously
4. **Market Impact**: Value and volume of MEV extraction

---

## 🚨 Troubleshooting

### **"Connection Error"**
- Try a different free RPC endpoint
- Check your internet connection
- Reduce `max_concurrent_requests` in config.json

### **"Too Many Requests"**
- Free RPCs have rate limits
- Reduce `max_concurrent_requests` to 5
- Add small delays between blocks

### **"No MEV Detected"**
- Collect more blocks (50-100+)
- MEV is not in every block
- Check recent DeFi-heavy blocks

---

## 💡 Next Steps

1. ✅ Collect initial data (10-50 blocks)
2. 📊 Review the HTML analysis report
3. 🔍 Look for interesting patterns
4. 📈 Collect more data for better insights
5. 🧠 (Optional) Train the ML model with more data

---

## 🆘 Need Help?

Check the main README.md for:
- Detailed documentation
- Architecture explanation
- Advanced features
- API documentation

---

**🎉 That's it! No API keys, no setup hassle. Just run and analyze!** 🚀
