import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataAnalyzer:
    """Class for analyzing collected cryptocurrency and MEV data"""
    
    def __init__(self, db_path: str = "crypto_data.db"):
        self.db_path = db_path
        self.tx_df = None
        self.mev_df = None
        self.blocks_df = None
        self._load_data()
    
    def _load_data(self):
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load tables
            self.tx_df = pd.read_sql_query('''
                SELECT * FROM transactions 
                ORDER BY block_number, transaction_index
            ''', conn)
            
            self.mev_df = pd.read_sql_query('''
                SELECT * FROM mev_opportunities 
                ORDER BY block_number, profit_usd DESC
            ''', conn)
            
            self.blocks_df = pd.read_sql_query('''
                SELECT * FROM blocks 
                ORDER BY block_number
            ''', conn)
            
            conn.close()
            
            # Convert timestamp columns
            if not self.tx_df.empty:
                self.tx_df['timestamp'] = pd.to_datetime(self.tx_df['timestamp'])
            if not self.mev_df.empty:
                self.mev_df['timestamp'] = pd.to_datetime(self.mev_df['timestamp'])
            if not self.blocks_df.empty:
                self.blocks_df['timestamp'] = pd.to_datetime(self.blocks_df['timestamp'])
                
            print(f"Loaded {len(self.tx_df)} transactions, {len(self.mev_df)} MEV opportunities, {len(self.blocks_df)} blocks")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Initialize empty DataFrames
            self.tx_df = pd.DataFrame()
            self.mev_df = pd.DataFrame()
            self.blocks_df = pd.DataFrame()
    
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        stats = {}
        
        if not self.tx_df.empty:
            stats['transactions'] = {
                'total_transactions': len(self.tx_df),
                'unique_blocks': self.tx_df['block_number'].nunique(),
                'avg_gas_price': self.tx_df['gas_price'].mean(),
                'total_value_eth': self.tx_df['value'].sum(),
                'avg_tx_per_block': len(self.tx_df) / self.tx_df['block_number'].nunique() if self.tx_df['block_number'].nunique() > 0 else 0
            }
        
        if not self.mev_df.empty:
            stats['mev'] = {
                'total_mev_opportunities': len(self.mev_df),
                'total_profit_usd': self.mev_df['profit_usd'].sum(),
                'avg_profit_usd': self.mev_df['profit_usd'].mean(),
                'mev_types': self.mev_df['mev_type'].value_counts().to_dict(),
                'blocks_with_mev': self.mev_df['block_number'].nunique(),
                'mev_percentage': (self.mev_df['block_number'].nunique() / self.tx_df['block_number'].nunique() * 100) if not self.tx_df.empty and self.tx_df['block_number'].nunique() > 0 else 0
            }
        
        return stats
    
    def plot_gas_price_distribution(self, save_path: str = None):
        """Plot gas price distribution"""
        if self.tx_df.empty:
            print("No transaction data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gas price histogram
        axes[0,0].hist(self.tx_df['gas_price'] / 1e9, bins=50, alpha=0.7)
        axes[0,0].set_xlabel('Gas Price (Gwei)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Gas Price Distribution')
        
        # Gas price over time
        if 'timestamp' in self.tx_df.columns:
            hourly_gas = self.tx_df.groupby(pd.Grouper(key='timestamp', freq='H'))['gas_price'].mean()
            axes[0,1].plot(hourly_gas.index, hourly_gas.values / 1e9)
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_ylabel('Average Gas Price (Gwei)')
            axes[0,1].set_title('Gas Price Over Time')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Transaction value distribution
        axes[1,0].hist(self.tx_df['value'], bins=50, alpha=0.7)
        axes[1,0].set_xlabel('Transaction Value (ETH)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Transaction Value Distribution')
        axes[1,0].set_yscale('log')
        
        # Gas used distribution
        axes[1,1].hist(self.tx_df['gas_used'], bins=50, alpha=0.7)
        axes[1,1].set_xlabel('Gas Used')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Gas Used Distribution')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_mev_analysis(self, save_path: str = None):
        """Plot MEV opportunity analysis"""
        if self.mev_df.empty:
            print("No MEV data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MEV types distribution
        mev_counts = self.mev_df['mev_type'].value_counts()
        axes[0,0].pie(mev_counts.values, labels=mev_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('MEV Types Distribution')
        
        # Profit distribution
        axes[0,1].hist(self.mev_df['profit_usd'], bins=30, alpha=0.7)
        axes[0,1].set_xlabel('Profit (USD)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('MEV Profit Distribution')
        axes[0,1].set_yscale('log')
        
        # Profit vs Gas Cost
        axes[1,0].scatter(self.mev_df['gas_cost_usd'], self.mev_df['profit_usd'], alpha=0.6)
        axes[1,0].set_xlabel('Gas Cost (USD)')
        axes[1,0].set_ylabel('Profit (USD)')
        axes[1,0].set_title('MEV Profit vs Gas Cost')
        axes[1,0].plot([0, self.mev_df['gas_cost_usd'].max()], [0, self.mev_df['gas_cost_usd'].max()], 'r--', alpha=0.5)
        
        # MEV by DEX
        dex_profits = self.mev_df.groupby('dex_name')['profit_usd'].sum().sort_values(ascending=True)
        axes[1,1].barh(range(len(dex_profits)), dex_profits.values)
        axes[1,1].set_yticks(range(len(dex_profits)))
        axes[1,1].set_yticklabels(dex_profits.index)
        axes[1,1].set_xlabel('Total Profit (USD)')
        axes[1,1].set_title('MEV Profit by DEX')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self) -> go.Figure:
        """Create an interactive Plotly dashboard"""
        if self.tx_df.empty:
            print("No data available for dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Gas Price Over Time', 'Transaction Volume', 'MEV Opportunities', 'Block Analysis'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Gas price over time
        if 'timestamp' in self.tx_df.columns:
            hourly_data = self.tx_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
                'gas_price': 'mean',
                'value': 'sum',
                'transaction_hash': 'count'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(x=hourly_data['timestamp'], y=hourly_data['gas_price']/1e9,
                          name='Avg Gas Price (Gwei)', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Transaction volume
        if not self.tx_df.empty:
            fig.add_trace(
                go.Scatter(x=self.tx_df['gas_price']/1e9, y=self.tx_df['value'],
                          mode='markers', name='Transactions',
                          marker=dict(size=4, opacity=0.6)),
                row=1, col=2
            )
        
        # MEV opportunities by type
        if not self.mev_df.empty:
            mev_counts = self.mev_df['mev_type'].value_counts()
            fig.add_trace(
                go.Bar(x=mev_counts.index, y=mev_counts.values, name='MEV Count'),
                row=2, col=1
            )
        
        # Block analysis
        if not self.tx_df.empty:
            block_stats = self.tx_df.groupby('block_number').agg({
                'gas_price': 'mean',
                'transaction_hash': 'count'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(x=block_stats['transaction_hash'], y=block_stats['gas_price']/1e9,
                          mode='markers', name='Blocks',
                          marker=dict(size=6, opacity=0.7)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Cryptocurrency Front-Running Analysis Dashboard",
            title_x=0.5
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Gas Price (Gwei)", row=1, col=1)
        
        fig.update_xaxes(title_text="Gas Price (Gwei)", row=1, col=2)
        fig.update_yaxes(title_text="Value (ETH)", row=1, col=2)
        
        fig.update_xaxes(title_text="MEV Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_xaxes(title_text="Transactions per Block", row=2, col=2)
        fig.update_yaxes(title_text="Avg Gas Price (Gwei)", row=2, col=2)
        
        return fig
    
    def detect_front_running_patterns(self) -> pd.DataFrame:
        """Detect potential front-running patterns in the data"""
        if self.tx_df.empty:
            print("No transaction data available")
            return pd.DataFrame()
        
        # Group transactions by block
        block_analysis = []
        
        for block_num in self.tx_df['block_number'].unique():
            block_txs = self.tx_df[self.tx_df['block_number'] == block_num].sort_values('transaction_index')
            
            if len(block_txs) < 3:
                continue
            
            # Look for gas price patterns
            gas_prices = block_txs['gas_price'].values
            
            # Detect potential sandwich attacks (high-low-high pattern)
            for i in range(len(gas_prices) - 2):
                if (gas_prices[i] > gas_prices[i+1] * 1.2 and 
                    gas_prices[i+2] > gas_prices[i+1] * 1.2):
                    
                    # Check if same sender for first and last transaction
                    tx1, tx2, tx3 = block_txs.iloc[i:i+3]['from_address'].values
                    if tx1 == tx3:
                        pattern = {
                            'block_number': block_num,
                            'pattern_type': 'sandwich_attack',
                            'frontrun_gas': gas_prices[i],
                            'victim_gas': gas_prices[i+1],
                            'backrun_gas': gas_prices[i+2],
                            'gas_ratio': gas_prices[i] / gas_prices[i+1],
                            'attacker_address': tx1,
                            'victim_address': tx2,
                            'estimated_profit': block_txs.iloc[i+1]['value'] * 0.005  # 0.5% estimate
                        }
                        block_analysis.append(pattern)
        
        return pd.DataFrame(block_analysis)
    
    def generate_report(self, output_file: str = "analysis_report.html"):
        """Generate a comprehensive HTML report"""
        stats = self.generate_summary_stats()
        patterns_df = self.detect_front_running_patterns()
        
        # Create interactive dashboard
        dashboard = self.create_interactive_dashboard()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Front-Running Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 20px 0; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffffcc; padding: 10px; border-left: 5px solid #ffeb3b; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cryptocurrency Front-Running Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
        """
        
        # Add transaction statistics
        if 'transactions' in stats:
            tx_stats = stats['transactions']
            html_content += f"""
                <h3>Transaction Statistics</h3>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Transactions</td><td>{tx_stats['total_transactions']:,}</td></tr>
                    <tr><td>Unique Blocks</td><td>{tx_stats['unique_blocks']:,}</td></tr>
                    <tr><td>Average Gas Price</td><td>{tx_stats['avg_gas_price']/1e9:.2f} Gwei</td></tr>
                    <tr><td>Total Value</td><td>{tx_stats['total_value_eth']:.2f} ETH</td></tr>
                    <tr><td>Avg Transactions per Block</td><td>{tx_stats['avg_tx_per_block']:.2f}</td></tr>
                </table>
            """
        
        # Add MEV statistics
        if 'mev' in stats:
            mev_stats = stats['mev']
            html_content += f"""
                <h3>MEV Statistics</h3>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total MEV Opportunities</td><td>{mev_stats['total_mev_opportunities']:,}</td></tr>
                    <tr><td>Total Profit</td><td>${mev_stats['total_profit_usd']:,.2f}</td></tr>
                    <tr><td>Average Profit</td><td>${mev_stats['avg_profit_usd']:,.2f}</td></tr>
                    <tr><td>Blocks with MEV</td><td>{mev_stats['blocks_with_mev']:,}</td></tr>
                    <tr><td>MEV Percentage</td><td>{mev_stats['mev_percentage']:.2f}%</td></tr>
                </table>
            """
        
        # Add detected patterns
        if not patterns_df.empty:
            html_content += f"""
                <div class="section">
                    <h2>Detected Front-Running Patterns</h2>
                    <div class="highlight">
                        <strong>Alert:</strong> Detected {len(patterns_df)} potential front-running patterns!
                    </div>
                    <table class="stats-table">
                        <tr><th>Block</th><th>Pattern Type</th><th>Gas Ratio</th><th>Estimated Profit</th></tr>
            """
            for _, row in patterns_df.head(10).iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['block_number']}</td>
                        <td>{row['pattern_type']}</td>
                        <td>{row['gas_ratio']:.2f}x</td>
                        <td>{row['estimated_profit']:.4f} ETH</td>
                    </tr>
                """
            html_content += "</table></div>"
        
        # Add dashboard
        if dashboard:
            dashboard_html = dashboard.to_html(include_plotlyjs='cdn')
            html_content += f"""
                <div class="section">
                    <h2>Interactive Dashboard</h2>
                    {dashboard_html}
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_file}")
        
        return stats, patterns_df

def main():
    """Main function to run analysis"""
    analyzer = DataAnalyzer()
    
    if analyzer.tx_df.empty:
        print("No data found. Please run the data collection script first.")
        return
    
    # Generate summary statistics
    print("=== CRYPTOCURRENCY FRONT-RUNNING ANALYSIS ===\n")
    
    stats = analyzer.generate_summary_stats()
    
    if 'transactions' in stats:
        tx_stats = stats['transactions']
        print("TRANSACTION STATISTICS:")
        print(f"  Total Transactions: {tx_stats['total_transactions']:,}")
        print(f"  Unique Blocks: {tx_stats['unique_blocks']:,}")
        print(f"  Average Gas Price: {tx_stats['avg_gas_price']/1e9:.2f} Gwei")
        print(f"  Total Value: {tx_stats['total_value_eth']:.2f} ETH")
        print(f"  Avg Transactions per Block: {tx_stats['avg_tx_per_block']:.2f}")
        print()
    
    if 'mev' in stats:
        mev_stats = stats['mev']
        print("MEV STATISTICS:")
        print(f"  Total MEV Opportunities: {mev_stats['total_mev_opportunities']:,}")
        print(f"  Total Profit: ${mev_stats['total_profit_usd']:,.2f}")
        print(f"  Average Profit: ${mev_stats['avg_profit_usd']:,.2f}")
        print(f"  MEV Percentage: {mev_stats['mev_percentage']:.2f}%")
        print()
    
    # Detect patterns
    patterns_df = analyzer.detect_front_running_patterns()
    if not patterns_df.empty:
        print("FRONT-RUNNING PATTERNS DETECTED:")
        print(f"  Found {len(patterns_df)} potential patterns")
        print(f"  Average gas ratio: {patterns_df['gas_ratio'].mean():.2f}x")
        print(f"  Total estimated profit: {patterns_df['estimated_profit'].sum():.4f} ETH")
        print()
    
    # Generate plots
    print("Generating analysis plots...")
    analyzer.plot_gas_price_distribution("gas_analysis.png")
    analyzer.plot_mev_analysis("mev_analysis.png")
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    analyzer.generate_report("front_running_analysis_report.html")
    
    print("Analysis complete! Check the generated files.")

if __name__ == "__main__":
    main()