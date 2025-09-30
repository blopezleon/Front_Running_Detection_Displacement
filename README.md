# Front-Running Detection with RAG PyTorch Model

A comprehensive system for collecting cryptocurrency transaction data and detecting front-running attacks using a Retrieval-Augmented Generation (RAG) PyTorch model.

## 🎯 Overview

This project provides tools to:
- Collect real-time cryptocurrency transaction data from multiple chains (Ethereum, Polygon, BSC, Arbitrum)
- Detect MEV (Maximal Extractable Value) opportunities including sandwich attacks, arbitrage, and liquidations
- Train a RAG-based PyTorch model for front-running detection
- Analyze patterns and generate comprehensive reports

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Collection│    │   RAG Model      │    │   Analysis &    │
│   (get_data.py)  ├───►│  (rag_model.py)  ├───►│   Reporting     │
│                 │    │                  │    │ (analyze_data.py)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SQLite DB     │    │  Knowledge Base  │    │   HTML Reports  │
│   - Transactions│    │  - FAISS Index   │    │   - Visualizations│
│   - MEV Data    │    │  - Embeddings    │    │   - Statistics  │
│   - Block Data  │    │  - Metadata      │    │   - Patterns    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

### Data Collection (`get_data.py`)
- **Multi-chain Support**: Ethereum, Polygon, BSC, Arbitrum
- **Real-time Data**: Latest block monitoring and historical data collection
- **MEV Detection**: Automatic detection of sandwich attacks, arbitrage, and liquidations
- **Concurrent Processing**: Async/await pattern for efficient data collection
- **Database Storage**: SQLite database for structured data storage

### RAG Model (`rag_model.py`)
- **Neural Architecture**: PyTorch-based model with transformer components
- **Knowledge Base**: FAISS vector database for pattern storage and retrieval
- **Feature Engineering**: Automated feature extraction from transaction data
- **Explainable AI**: RAG approach provides explanations for predictions
- **Transfer Learning**: Pre-trained transformer models for text understanding

### Analysis & Visualization (`analyze_data.py`)
- **Statistical Analysis**: Comprehensive transaction and MEV statistics
- **Pattern Detection**: Algorithmic front-running pattern identification
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **HTML Reports**: Automated report generation with charts and insights

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Front_Running_Detection_Displacement
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Settings
Edit `config.json` to add your API keys and adjust settings:
```json
{
  "ethereum_rpc": "https://eth.llamarpc.com",
  "dune_api_key": "your_dune_api_key",
  "etherscan_api_key": "your_etherscan_api_key",
  "target_tokens": ["0xA0b86a33E6441E359e0DC2Db18db4ac7F08A2056"],
  "max_concurrent_requests": 10
}
```

## 🔧 Usage

### Step 1: Collect Data
```bash
python get_data.py
```
This will:
- Connect to blockchain RPCs
- Collect transaction data from recent blocks
- Detect MEV opportunities
- Store data in `crypto_data.db`

### Step 2: Train RAG Model
```python
from rag_model import FrontRunningRAGSystem

# Initialize system
rag_system = FrontRunningRAGSystem()

# Prepare knowledge base from collected data
rag_system.prepare_knowledge_base()

# Train the model
rag_system.train_model()

# Save trained model
rag_system.save_model()
```

### Step 3: Analyze Data
```bash
python analyze_data.py
```
This generates:
- Statistical summaries
- Pattern detection results
- Interactive visualizations
- HTML report (`front_running_analysis_report.html`)

### Step 4: Detect Front-Running
```python
from rag_model import FrontRunningRAGSystem

# Load trained model
rag_system = FrontRunningRAGSystem()
rag_system.load_model("rag_model")

# Analyze new block data
block_features = {
    'gas_price_mean': 50e9,
    'gas_price_std': 10e9,
    'tx_count': 48,
    'total_value': 100.5
}

result = rag_system.detect_front_running(block_features)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanation: {result['explanation']}")
```

## 📊 Data Schema

### Transactions Table
```sql
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    block_number INTEGER,
    transaction_hash TEXT UNIQUE,
    from_address TEXT,
    to_address TEXT,
    value REAL,
    gas_price INTEGER,
    gas_used INTEGER,
    timestamp DATETIME,
    chain_id INTEGER
);
```

### MEV Opportunities Table
```sql
CREATE TABLE mev_opportunities (
    id INTEGER PRIMARY KEY,
    block_number INTEGER,
    mev_type TEXT,  -- 'sandwich', 'arbitrage', 'liquidation'
    profit_usd REAL,
    gas_cost_usd REAL,
    victim_tx_hash TEXT,
    frontrun_tx_hash TEXT,
    dex_name TEXT,
    timestamp DATETIME
);
```

## 🧠 RAG Model Architecture

The RAG (Retrieval-Augmented Generation) model combines:

1. **Feature Encoder**: Neural network for numerical transaction features
2. **Text Encoder**: Pre-trained transformer for transaction descriptions  
3. **Knowledge Base**: FAISS vector database storing MEV patterns
4. **Retrieval System**: Semantic search for relevant historical patterns
5. **Fusion Layer**: Combines retrieved context with current features
6. **Classifier**: Binary classification (front-running vs normal)

### Model Components
```python
class RAGModel(nn.Module):
    def __init__(self, config):
        self.feature_encoder = nn.Sequential(...)  # Numerical features
        self.text_encoder = AutoModel.from_pretrained(...)  # Text features
        self.knowledge_base = KnowledgeBase()  # Vector retrieval
        self.fusion = nn.Sequential(...)  # Feature fusion
        self.classifier = nn.Linear(...)  # Final classification
```

## 📈 Detection Methods

### Sandwich Attack Detection
- Identifies high-gas → low-gas → high-gas patterns
- Checks for same attacker address in front/back transactions
- Analyzes DEX interaction patterns
- Estimates profit based on victim transaction value

### Arbitrage Detection  
- Finds transactions interacting with multiple DEXes
- Analyzes price differences across exchanges
- Detects flash loan patterns
- Calculates estimated arbitrage profits

### Liquidation Detection
- Monitors lending protocol interactions
- Identifies liquidation function calls
- Tracks collateral seizures
- Estimates liquidation bonuses

## 🔍 Pattern Analysis

The system automatically detects:
- **Gas Price Anomalies**: Unusual gas price spikes
- **Transaction Ordering**: Suspicious transaction sequences
- **Value Transfers**: Large value movements
- **Contract Interactions**: DEX and DeFi protocol usage
- **Timing Patterns**: Front-running timing analysis

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Web3.py 6.0+
- Transformers 4.30+
- Pandas 2.0+
- NumPy 1.24+
- FAISS-CPU 1.7+
- SQLite3

## ⚠️ Limitations & Disclaimers

1. **Data Sources**: Relies on public RPC endpoints (rate limits apply)
2. **Detection Accuracy**: Heuristic-based detection may have false positives/negatives
3. **Real-time Performance**: Processing delays may affect real-time detection
4. **Legal Compliance**: Ensure compliance with local regulations
5. **Educational Purpose**: This tool is for research and educational use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-detection-method`)
3. Commit changes (`git commit -am 'Add new detection method'`)
4. Push to branch (`git push origin feature/new-detection-method`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Ethereum Documentation](https://ethereum.org/developers/)
- [MEV Research](https://ethereum.org/en/developers/docs/mev/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [FAISS Documentation](https://faiss.ai/)

## 📞 Support

For questions, issues, or contributions, please:
1. Check existing [Issues](../../issues)
2. Create a new issue with detailed description
3. Join our [Discord community](#) (if available)

---

**⚡ Happy front-running detection!** 🚀