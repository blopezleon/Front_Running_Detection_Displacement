#!/usr/bin/env python3
"""
Full Pipeline - Automated Front-Running Detection System
Collects data, trains RAG model, and performs detection
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_full_pipeline():
    """Run the complete front-running detection pipeline"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          🚀 FRONT-RUNNING DETECTION SYSTEM - FULL PIPELINE 🚀               ║
║                                                                              ║
║  This script will:                                                          ║
║  1. Collect Ethereum transaction data from the blockchain                   ║
║  2. Prepare and train the RAG (Retrieval-Augmented Generation) model       ║
║  3. Run front-running detection on collected data                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    try:
        # Step 1: Collect Data
        logger.info("=" * 80)
        logger.info("STEP 1: COLLECTING ETHEREUM DATA")
        logger.info("=" * 80)
        
        from get_data import CryptoDataCollector
        
        collector = CryptoDataCollector()
        
        # Check current database stats
        stats = collector.get_database_stats()
        if stats['total_blocks'] > 0:
            logger.info(f"📦 Existing data found:")
            logger.info(f"   Blocks: {stats['total_blocks']}")
            logger.info(f"   Transactions: {stats['total_transactions']:,}")
            logger.info(f"   MEV opportunities: {stats['mev_opportunities']}")
        
        # Collect 20 blocks of data (enough for training)
        logger.info(f"\n🔄 Collecting 20 new blocks from Ethereum mainnet...")
        await collector.collect_latest_data(num_blocks=20, chain_id=1)
        
        # Export data for training
        logger.info("\n💾 Exporting data for training...")
        file_paths = collector.export_data_for_training()
        
        await collector.close()
        
        logger.info("\n✅ Data collection completed!")
        logger.info(f"   Files created:")
        for key, path in file_paths.items():
            logger.info(f"   • {path}")
        
        # Step 2: Train RAG Model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: TRAINING RAG MODEL")
        logger.info("=" * 80)
        
        from rag_model import FrontRunningRAGSystem, RAGConfig
        
        # Initialize RAG system
        config = RAGConfig()
        config.epochs = 5  # Reduced for demo
        config.batch_size = 16
        
        logger.info(f"\n🧠 Initializing RAG model...")
        logger.info(f"   Device: {config.device}")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   Embedding: {config.embedding_model}")
        
        rag_system = FrontRunningRAGSystem(config)
        
        # Prepare knowledge base
        logger.info(f"\n📚 Preparing knowledge base from collected data...")
        rag_system.prepare_knowledge_base()
        
        # Train model
        logger.info(f"\n🎯 Training model ({config.epochs} epochs)...")
        logger.info("   This may take a few minutes...")
        rag_system.train_model()
        
        # Save model
        logger.info(f"\n💾 Saving trained model...")
        rag_system.save_model("rag_model")
        
        logger.info("\n✅ RAG model training completed!")
        
        # Step 3: Run Detection
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: RUNNING FRONT-RUNNING DETECTION")
        logger.info("=" * 80)
        
        # Test detection on sample block features
        logger.info(f"\n🔍 Testing detection on sample transaction patterns...")
        
        test_cases = [
            {
                'name': 'Normal Trading Block',
                'features': {
                    'gas_price_mean': 30e9,
                    'gas_price_std': 5e9,
                    'gas_price_min': 25e9,
                    'gas_price_max': 40e9,
                    'total_value': 50.5,
                    'avg_value': 1.5,
                    'value_std': 2.1,
                    'total_gas_used': 8000000,
                    'avg_gas_used': 150000,
                    'tx_count': 53
                }
            },
            {
                'name': 'Suspicious High Gas Variation',
                'features': {
                    'gas_price_mean': 80e9,
                    'gas_price_std': 50e9,
                    'gas_price_min': 30e9,
                    'gas_price_max': 500e9,
                    'total_value': 250.8,
                    'avg_value': 8.5,
                    'value_std': 25.3,
                    'total_gas_used': 15000000,
                    'avg_gas_used': 350000,
                    'tx_count': 42
                }
            },
            {
                'name': 'Potential MEV Block',
                'features': {
                    'gas_price_mean': 150e9,
                    'gas_price_std': 75e9,
                    'gas_price_min': 50e9,
                    'gas_price_max': 800e9,
                    'total_value': 500.2,
                    'avg_value': 15.6,
                    'value_std': 45.8,
                    'total_gas_used': 20000000,
                    'avg_gas_used': 450000,
                    'tx_count': 38
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n📊 Test Case {i}: {test_case['name']}")
            logger.info("   " + "-" * 60)
            
            result = rag_system.detect_front_running(test_case['features'])
            
            prediction_label = "🚨 FRONT-RUNNING DETECTED" if result['prediction'] == 1 else "✅ NORMAL TRADING"
            logger.info(f"   Prediction: {prediction_label}")
            logger.info(f"   Confidence: {result['confidence']:.2%}")
            logger.info(f"   Explanation: {result['explanation']}")
        
        # Step 4: Analyze Data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: GENERATING ANALYSIS REPORT")
        logger.info("=" * 80)
        
        from analyze_data import DataAnalyzer
        
        analyzer = DataAnalyzer()
        
        if not analyzer.tx_df.empty:
            # Generate statistics
            stats = analyzer.generate_summary_stats()
            
            logger.info("\n📈 SUMMARY STATISTICS:")
            
            if 'transactions' in stats:
                tx_stats = stats['transactions']
                logger.info("\n💰 TRANSACTION STATISTICS:")
                logger.info(f"   Total Transactions: {tx_stats['total_transactions']:,}")
                logger.info(f"   Unique Blocks: {tx_stats['unique_blocks']:,}")
                logger.info(f"   Average Gas Price: {tx_stats['avg_gas_price']/1e9:.2f} Gwei")
                logger.info(f"   Total Value: {tx_stats['total_value_eth']:.4f} ETH")
                logger.info(f"   Avg TX per Block: {tx_stats['avg_tx_per_block']:.1f}")
            
            if 'mev' in stats and stats['mev']['total_mev_opportunities'] > 0:
                mev_stats = stats['mev']
                logger.info("\n🎯 MEV STATISTICS:")
                logger.info(f"   MEV Opportunities: {mev_stats['total_mev_opportunities']:,}")
                logger.info(f"   Total Profit: ${mev_stats['total_profit_usd']:,.2f}")
                logger.info(f"   Average Profit: ${mev_stats['avg_profit_usd']:,.2f}")
                logger.info(f"   MEV Percentage: {mev_stats['mev_percentage']:.2f}%")
            
            # Detect patterns
            logger.info("\n🔍 Detecting front-running patterns...")
            patterns_df = analyzer.detect_front_running_patterns()
            
            if not patterns_df.empty:
                logger.info(f"\n🚨 FRONT-RUNNING PATTERNS DETECTED: {len(patterns_df)}")
                logger.info(f"   Average gas ratio: {patterns_df['gas_ratio'].mean():.2f}x")
                logger.info(f"   Total estimated profit: {patterns_df['estimated_profit'].sum():.4f} ETH")
            else:
                logger.info("\n✅ No obvious front-running patterns detected")
            
            # Generate HTML report
            logger.info("\n📋 Generating HTML report...")
            analyzer.generate_report("front_running_detection_report.html")
            logger.info("✅ Report saved: front_running_detection_report.html")
        
        # Final Summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        logger.info("\n📂 Generated Files:")
        logger.info(f"   • rag_model/              - Trained RAG model")
        logger.info(f"   • crypto_data.db           - Transaction database")
        logger.info(f"   • training_data/           - Exported training data")
        logger.info(f"   • front_running_detection_report.html  - Analysis report")
        
        logger.info("\n🔧 Next Steps:")
        logger.info("   1. Open the HTML report to view detailed analysis")
        logger.info("   2. Run more data collection to improve model accuracy")
        logger.info("   3. Use the trained model for real-time detection")
        
        logger.info("\n💡 To use the trained model:")
        logger.info("   from rag_model import FrontRunningRAGSystem")
        logger.info("   rag = FrontRunningRAGSystem()")
        logger.info("   rag.load_model('rag_model')")
        logger.info("   result = rag.detect_front_running(your_block_features)")
        
    except Exception as e:
        logger.error(f"\n❌ Error during pipeline execution: {e}")
        logger.error(f"   Details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(run_full_pipeline())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

