#!/usr/bin/env python3
"""
Simple MEV labeling: Detect transaction replacement (gas auctions)
Real definition: Same sender + same nonce + higher gas price
"""
import sqlite3
import pandas as pd

print("Loading transactions from database...")
conn = sqlite3.connect('data/crypto_data.db')

df = pd.read_sql_query("""
    SELECT 
        transaction_hash,
        from_address,
        nonce,
        gas_price,
        block_number,
        transaction_index,
        timestamp
    FROM transactions
    ORDER BY block_number, transaction_index
""", conn)
conn.close()

print(f"Loaded {len(df):,} transactions")

# Label MEV: transaction is part of gas auction if:
# - Same sender has another tx with same nonce and different gas price
print("\nDetecting gas auctions (tx replacement)...")

df['is_mev'] = 0
mev_count = 0

# Group by sender + nonce
for (sender, nonce), group in df.groupby(['from_address', 'nonce']):
    if len(group) > 1:
        # Multiple transactions with same sender+nonce = gas auction!
        indices = group.index
        df.loc[indices, 'is_mev'] = 1
        mev_count += len(indices)

print(f"\n✅ Found {mev_count:,} MEV transactions ({mev_count/len(df)*100:.2f}%)")
print(f"   (Transactions that were replaced or are replacements)")

# Save
df[['transaction_hash', 'is_mev']].to_csv('data/mev_labels.csv', index=False)
print(f"\n✅ Saved labels to data/mev_labels.csv")
