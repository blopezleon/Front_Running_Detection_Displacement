#!/usr/bin/env python3
"""
Quick Status Check

Shows current status of data collection and labeling.
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("SYSTEM STATUS")
    print("="*70 + "\n")
    
    # Check database
    db_path = "crypto_data.db"
    if not os.path.exists(db_path):
        print("❌ Database not found: crypto_data.db")
        print("   Run: python collect_data.py")
        return
    
    print("✓ Database: crypto_data.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Block stats
        cursor.execute("SELECT COUNT(DISTINCT block_number) FROM transactions")
        blocks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transactions")
        txs = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(block_number), MAX(block_number) FROM transactions")
        block_range = cursor.fetchone()
        
        print(f"  • Blocks collected: {blocks:,}")
        print(f"  • Transactions: {txs:,}")
        if block_range and block_range[0]:
            print(f"  • Block range: {block_range[0]:,} - {block_range[1]:,}")
            print(f"  • Coverage: {block_range[1] - block_range[0] + 1:,} block span")
        
        conn.close()
        
    except Exception as e:
        print(f"  ⚠ Error reading database: {e}")
    
    print()
    
    # Check labels
    csv_path = "labeled_training_data.csv"
    if not os.path.exists(csv_path):
        print("⚠ No labeled data yet")
        print("   Run: python label_data.py --monitor")
        print()
        return
    
    print("✓ Labeled Data: labeled_training_data.csv")
    
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print("  • No patterns detected yet")
        else:
            print(f"  • Total patterns: {len(df):,}")
            print(f"  • Blocks with MEV: {df['block_number'].nunique():,}")
            print(f"  • Detection rate: {len(df)/txs*100:.3f}% of transactions")
            
            if 'is_front_running' in df.columns:
                sandwich_count = df['is_front_running'].sum()
                print(f"  • Sandwich attacks: {sandwich_count:,}")
        
    except Exception as e:
        print(f"  ⚠ Error reading labels: {e}")
    
    print()
    print("="*70)
    print("QUICK ACTIONS")
    print("="*70)
    print()
    print("Start continuous collection:")
    print("  → python collect_data.py")
    print()
    print("Start continuous labeling:")
    print("  → python label_data.py --monitor")
    print()
    print("Start both together:")
    print("  → Terminal 1: python collect_data.py")
    print("  → Terminal 2: python label_data.py --monitor")
    print()
    print("View detailed guide:")
    print("  → python HOW_TO_RUN.py")
    print()
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
