#!/bin/bash
# Setup Script for Flash Boys 2.0 MEV Detection

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        Flash Boys 2.0 MEV Detection - Setup                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "1. Checking Python version..."
python --version || python3 --version
echo ""

# Install dependencies
echo "2. Installing dependencies..."
pip install web3 aiohttp pandas numpy
echo ""

# Create data directory
echo "3. Creating data directory..."
mkdir -p data
echo "   ✅ data/ directory created"
echo ""

# Check if data exists
if [ -f "data/crypto_data.db" ]; then
    echo "4. Checking existing data..."
    BLOCKS=$(sqlite3 data/crypto_data.db "SELECT COUNT(DISTINCT block_number) FROM transactions" 2>/dev/null || echo "0")
    TXS=$(sqlite3 data/crypto_data.db "SELECT COUNT(*) FROM transactions" 2>/dev/null || echo "0")
    echo "   ✅ Found existing data: $BLOCKS blocks, $TXS transactions"
else
    echo "4. No existing data found"
    echo "   → Run: python collect_data.py"
fi
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                      SETUP COMPLETE                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "   1. Collect data (optional if you already have data):"
echo "      python collect_data.py"
echo ""
echo "   2. Analyze with Flash Boys algorithm:"
echo "      python flashboys_analysis.py"
echo ""
echo "   3. Train your model:"
echo "      Use data/flashboys_analysis.csv"
echo ""
echo "📖 Read WORKFLOW.md for complete guide"
echo ""
