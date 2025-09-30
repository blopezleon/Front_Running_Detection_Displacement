#!/usr/bin/env python3
"""
Example script demonstrating the front-running detection system usage.
This script shows how to collect data, train the model, and perform detection.
"""

import asyncio
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔍 FRONT-RUNNING DETECTION SYSTEM 🔍                     ║
║                                                                              ║
║   A comprehensive RAG-based PyTorch model for detecting crypto front-runs   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def collect_sample_data():
    """Collect sample data for demonstration"""
    logger.info("🔄 Starting data collection...")
    
    try:
        from get_data import CryptoDataCollector
        
        # Initialize collector
        collector = CryptoDataCollector()
        
        # Collect data from latest 3 blocks (small sample for demo)
        await collector.collect_latest_data(num_blocks=3, chain_id=1)
        
        # Export data for training
        file_paths = collector.export_data_for_training()
        
        # Clean up
        await collector.close()
        
        logger.info("✅ Data collection completed successfully!")
        logger.info(f"📊 Exported files: {list(file_paths.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during data collection: {e}")
        return False

def train_rag_model():
    """Train the RAG model with collected data"""
    logger.info("🧠 Starting RAG model training...")
    
    try:
        # Check if we have the required dependencies
        import torch
        logger.info(f"🔥 PyTorch version: {torch.__version__}")
        logger.info(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        from rag_model import FrontRunningRAGSystem, RAGConfig
        
        # Initialize system with configuration
        config = RAGConfig(
            epochs=3,  # Reduced for demo
            batch_size=16,
            learning_rate=1e-4
        )
        
        rag_system = FrontRunningRAGSystem(config)
        
        # Check if database exists
        if not Path("crypto_data.db").exists():
            logger.warning("⚠️  No database found. Please run data collection first.")
            return False
        
        # Prepare knowledge base
        logger.info("📚 Building knowledge base...")
        rag_system.prepare_knowledge_base()
        
        # Train model
        logger.info("🏋️  Training model...")
        rag_system.train_model()
        
        # Save model
        logger.info("💾 Saving trained model...")
        rag_system.save_model("demo_rag_model")
        
        logger.info("✅ Model training completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.error("💡 Please install requirements: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        return False

def analyze_collected_data():
    """Analyze the collected data"""
    logger.info("📈 Starting data analysis...")
    
    try:
        from analyze_data import DataAnalyzer
        
        # Initialize analyzer
        analyzer = DataAnalyzer()
        
        if analyzer.tx_df.empty:
            logger.warning("⚠️  No transaction data found for analysis.")
            return False
        
        # Generate statistics
        stats = analyzer.generate_summary_stats()
        
        # Print summary
        logger.info("📊 ANALYSIS RESULTS:")
        
        if 'transactions' in stats:
            tx_stats = stats['transactions']
            logger.info(f"   💰 Total Transactions: {tx_stats['total_transactions']:,}")
            logger.info(f"   🏗️  Unique Blocks: {tx_stats['unique_blocks']:,}")
            logger.info(f"   ⛽ Average Gas Price: {tx_stats['avg_gas_price']/1e9:.2f} Gwei")
            logger.info(f"   💎 Total Value: {tx_stats['total_value_eth']:.4f} ETH")
        
        if 'mev' in stats:
            mev_stats = stats['mev']
            logger.info(f"   🎯 MEV Opportunities: {mev_stats['total_mev_opportunities']:,}")
            logger.info(f"   💸 Total MEV Profit: ${mev_stats['total_profit_usd']:,.2f}")
            logger.info(f"   📊 MEV Percentage: {mev_stats['mev_percentage']:.2f}%")
        
        # Detect patterns
        patterns_df = analyzer.detect_front_running_patterns()
        if not patterns_df.empty:
            logger.info(f"   🚨 Front-running Patterns Detected: {len(patterns_df)}")
            avg_profit = patterns_df['estimated_profit'].sum()
            logger.info(f"   💰 Estimated Total Profit: {avg_profit:.4f} ETH")
        
        # Generate comprehensive report
        logger.info("📋 Generating comprehensive report...")
        analyzer.generate_report("demo_analysis_report.html")
        
        logger.info("✅ Analysis completed successfully!")
        logger.info("📄 Report saved as: demo_analysis_report.html")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies for analysis: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        return False

def demonstrate_detection():
    """Demonstrate front-running detection on sample data"""
    logger.info("🔍 Demonstrating front-running detection...")
    
    try:
        from rag_model import FrontRunningRAGSystem
        
        # Check if trained model exists
        if not Path("demo_rag_model").exists():
            logger.warning("⚠️  No trained model found. Training basic model first...")
            if not train_rag_model():
                return False
        
        # Load system
        rag_system = FrontRunningRAGSystem()
        rag_system.load_model("demo_rag_model")
        
        # Test with sample block features
        test_cases = [
            {
                "name": "Normal Block",
                "features": {
                    'gas_price_mean': 20e9,
                    'gas_price_std': 5e9,
                    'gas_price_min': 15e9,
                    'gas_price_max': 30e9,
                    'total_value': 10.5,
                    'avg_value': 0.21,
                    'value_std': 1.2,
                    'total_gas_used': 8000000,
                    'avg_gas_used': 160000,
                    'tx_count': 50
                }
            },
            {
                "name": "Suspicious Block (High Gas Variance)",
                "features": {
                    'gas_price_mean': 100e9,
                    'gas_price_std': 50e9,
                    'gas_price_min': 20e9,
                    'gas_price_max': 500e9,  # Very high max
                    'total_value': 150.0,
                    'avg_value': 3.0,
                    'value_std': 10.5,
                    'total_gas_used': 12000000,
                    'avg_gas_used': 240000,
                    'tx_count': 50
                }
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\n🧪 Testing: {test_case['name']}")
            
            result = rag_system.detect_front_running(test_case['features'])
            
            prediction_text = "🚨 FRONT-RUNNING DETECTED" if result['prediction'] == 1 else "✅ NORMAL TRADING"
            
            logger.info(f"   📊 Prediction: {prediction_text}")
            logger.info(f"   🎯 Confidence: {result['confidence']:.1%}")
            logger.info(f"   💭 Explanation: {result['explanation'][:100]}...")
        
        logger.info("\n✅ Detection demonstration completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during detection demonstration: {e}")
        return False

def main():
    """Main function to run the demonstration"""
    print_banner()
    
    logger.info("🚀 Starting Front-Running Detection System Demo")
    logger.info("=" * 60)
    
    # Step 1: Collect sample data
    logger.info("STEP 1: Data Collection")
    if not asyncio.run(collect_sample_data()):
        logger.error("❌ Demo failed at data collection step")
        return
    
    time.sleep(1)
    
    # Step 2: Analyze collected data
    logger.info("\nSTEP 2: Data Analysis")
    if not analyze_collected_data():
        logger.error("❌ Demo failed at analysis step")
        return
    
    time.sleep(1)
    
    # Step 3: Train RAG model (commented out for speed in demo)
    # Uncomment this if you want to train the full model
    # logger.info("\nSTEP 3: Model Training")
    # if not train_rag_model():
    #     logger.error("❌ Demo failed at training step")
    #     return
    
    # Step 4: Demonstrate detection
    logger.info("\nSTEP 4: Front-Running Detection Demo")
    demonstrate_detection()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 DEMO COMPLETED SUCCESSFULLY!")
    logger.info("\n📋 Generated Files:")
    logger.info("   • crypto_data.db - Transaction database")
    logger.info("   • demo_analysis_report.html - Analysis report")
    logger.info("   • training_data/ - Exported training data")
    logger.info("   • demo_rag_model/ - Trained RAG model (if training was run)")
    
    logger.info("\n🔧 Next Steps:")
    logger.info("   1. Review the analysis report in your browser")
    logger.info("   2. Modify config.json with your API keys")
    logger.info("   3. Run with more blocks for better results")
    logger.info("   4. Train the full model for production use")
    
    logger.info("\n💡 Tips:")
    logger.info("   • Use python get_data.py for data collection")
    logger.info("   • Use python analyze_data.py for analysis")
    logger.info("   • Check README.md for detailed instructions")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        logger.error("Please check your installation and try again.")