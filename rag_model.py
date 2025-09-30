import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import sqlite3
from pathlib import Path
import faiss
from dataclasses import dataclass
import pickle
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG model"""
    model_name: str = "microsoft/DialoGPT-medium"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_seq_length: int = 512
    embedding_dim: int = 384
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TransactionDataset(Dataset):
    """Dataset class for transaction data"""
    
    def __init__(self, 
                 transactions_df: pd.DataFrame,
                 mev_df: pd.DataFrame,
                 tokenizer,
                 config: RAGConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        # Prepare features and labels
        self.features, self.labels, self.texts = self._prepare_data(transactions_df, mev_df)
        
    def _prepare_data(self, tx_df: pd.DataFrame, mev_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training"""
        
        # Group transactions by block
        block_features = tx_df.groupby('block_number').agg({
            'gas_price': ['mean', 'std', 'min', 'max'],
            'value': ['sum', 'mean', 'std'],
            'gas_used': ['sum', 'mean'],
            'transaction_hash': 'count'
        }).reset_index()
        
        # Flatten column names
        block_features.columns = ['block_number', 'gas_price_mean', 'gas_price_std', 
                                'gas_price_min', 'gas_price_max', 'total_value', 
                                'avg_value', 'value_std', 'total_gas_used', 
                                'avg_gas_used', 'tx_count']
        
        # Create text descriptions for each block
        texts = []
        for _, row in block_features.iterrows():
            text = f"Block {row['block_number']} has {row['tx_count']} transactions. " \
                   f"Average gas price: {row['gas_price_mean']:.2e}, " \
                   f"Total value: {row['total_value']:.4f} ETH, " \
                   f"Gas used: {row['total_gas_used']}"
            texts.append(text)
        
        # Add MEV labels
        mev_blocks = mev_df.groupby('block_number').agg({
            'profit_usd': 'sum',
            'mev_type': lambda x: ','.join(x.unique())
        }).reset_index()
        
        # Merge with block features
        merged_df = block_features.merge(mev_blocks, on='block_number', how='left')
        merged_df['has_mev'] = merged_df['profit_usd'].notna().astype(int)
        merged_df['profit_usd'] = merged_df['profit_usd'].fillna(0)
        
        # Normalize features
        feature_cols = ['gas_price_mean', 'gas_price_std', 'gas_price_min', 'gas_price_max',
                       'total_value', 'avg_value', 'value_std', 'total_gas_used', 'avg_gas_used', 'tx_count']
        
        scaler = StandardScaler()
        features = scaler.fit_transform(merged_df[feature_cols])
        labels = merged_df['has_mev'].values
        
        return features, labels, texts
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        # Tokenize text
        text_encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
        
        return {
            'features': feature,
            'labels': label,
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'text': self.texts[idx]
        }

class KnowledgeBase:
    """Vector database for storing and retrieving transaction patterns"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        if metadata is None:
            metadata = [{}] * len(documents)
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Initialize or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        if self.index is None:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
                
        return results
    
    def save(self, path: str):
        """Save knowledge base to disk"""
        Path(path).mkdir(exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, f"{path}/faiss_index.idx")
        
        # Save documents and metadata
        with open(f"{path}/documents.json", 'w') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f, indent=2)
            
    def load(self, path: str):
        """Load knowledge base from disk"""
        # Load FAISS index
        index_path = f"{path}/faiss_index.idx"
        if Path(index_path).exists():
            self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        docs_path = f"{path}/documents.json"
        if Path(docs_path).exists():
            with open(docs_path, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']

class RAGModel(nn.Module):
    """RAG model for front-running detection"""
    
    def __init__(self, config: RAGConfig):
        super().__init__()
        self.config = config
        
        # Feature encoder (for numerical transaction features)
        self.feature_encoder = nn.Sequential(
            nn.Linear(10, config.hidden_dim),  # 10 numerical features
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Text encoder (for transaction descriptions)
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + self.text_encoder.config.hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(config.hidden_dim, 2)  # Binary classification
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase(config.embedding_model)
        
    def forward(self, features, input_ids, attention_mask, retrieve_context=True):
        # Encode numerical features
        feature_encoded = self.feature_encoder(features)
        
        # Encode text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_encoded = text_outputs.last_hidden_state[:, 0, :]  # Use CLS token
        
        # Fusion
        combined = torch.cat([feature_encoded, text_encoded], dim=-1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'features': feature_encoded,
            'text_features': text_encoded,
            'fused_features': fused
        }
    
    def retrieve_and_generate(self, query: str, features: torch.Tensor) -> Dict[str, Any]:
        """Retrieve relevant context and generate explanation"""
        # Retrieve relevant documents from knowledge base
        retrieved_docs = self.knowledge_base.search(query, top_k=3)
        
        # Create context from retrieved documents
        context = ""
        if retrieved_docs:
            context = " ".join([doc['document'] for doc in retrieved_docs])
        
        # Forward pass with current input
        with torch.no_grad():
            outputs = self.forward(
                features.unsqueeze(0), 
                torch.LongTensor([[0]]),  # Dummy input_ids
                torch.LongTensor([[1]])   # Dummy attention_mask
            )
        
        # Generate prediction and explanation
        prediction = torch.softmax(outputs['logits'], dim=-1)
        predicted_class = torch.argmax(prediction, dim=-1).item()
        confidence = torch.max(prediction, dim=-1)[0].item()
        
        explanation = f"Based on the transaction features and similar patterns in the knowledge base, "
        explanation += f"this appears to be {'front-running' if predicted_class == 1 else 'normal trading'} "
        explanation += f"with {confidence:.2%} confidence."
        
        if retrieved_docs:
            explanation += f"\n\nSimilar patterns found:\n"
            for i, doc in enumerate(retrieved_docs[:2], 1):
                explanation += f"{i}. {doc['document'][:100]}...\n"
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'explanation': explanation,
            'retrieved_context': retrieved_docs
        }

class RAGTrainer:
    """Trainer class for RAG model"""
    
    def __init__(self, model: RAGModel, config: RAGConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].squeeze().to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(features, input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].squeeze().to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(features, input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), 'best_rag_model.pt')
                logger.info("Saved new best model")
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_metrics['loss']:.4f}, Val Acc = {val_metrics['accuracy']:.4f}")

class FrontRunningRAGSystem:
    """Main system for front-running detection with RAG"""
    
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = RAGModel(config)
        self.trainer = RAGTrainer(self.model, config)
        
    def prepare_knowledge_base(self, db_path: str = "crypto_data.db"):
        """Prepare knowledge base from collected data"""
        conn = sqlite3.connect(db_path)
        
        # Load MEV opportunities for knowledge base
        mev_df = pd.read_sql_query('''
            SELECT * FROM mev_opportunities 
            ORDER BY profit_usd DESC
        ''', conn)
        
        documents = []
        metadata = []
        
        for _, row in mev_df.iterrows():
            doc = f"MEV {row['mev_type']} attack in block {row['block_number']} "
            doc += f"generated ${row['profit_usd']:.2f} profit on {row['dex_name']} DEX. "
            doc += f"Gas cost: ${row['gas_cost_usd']:.2f}, Net profit: ${row['net_profit_usd']:.2f}."
            
            documents.append(doc)
            metadata.append({
                'block_number': row['block_number'],
                'mev_type': row['mev_type'],
                'profit_usd': row['profit_usd'],
                'dex_name': row['dex_name']
            })
        
        self.model.knowledge_base.add_documents(documents, metadata)
        conn.close()
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def train_model(self, db_path: str = "crypto_data.db"):
        """Train the RAG model"""
        conn = sqlite3.connect(db_path)
        
        # Load data
        tx_df = pd.read_sql_query('SELECT * FROM transactions', conn)
        mev_df = pd.read_sql_query('SELECT * FROM mev_opportunities', conn)
        conn.close()
        
        if tx_df.empty:
            logger.error("No transaction data found. Please collect data first.")
            return
        
        # Create dataset
        dataset = TransactionDataset(tx_df, mev_df, self.tokenizer, self.config)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Train model
        self.trainer.train(train_loader, val_loader)
        
        logger.info("Training completed!")
    
    def detect_front_running(self, block_features: Dict) -> Dict[str, Any]:
        """Detect front-running in new transaction data"""
        # Convert features to tensor
        feature_vector = np.array([
            block_features.get('gas_price_mean', 0),
            block_features.get('gas_price_std', 0),
            block_features.get('gas_price_min', 0),
            block_features.get('gas_price_max', 0),
            block_features.get('total_value', 0),
            block_features.get('avg_value', 0),
            block_features.get('value_std', 0),
            block_features.get('total_gas_used', 0),
            block_features.get('avg_gas_used', 0),
            block_features.get('tx_count', 0)
        ])
        
        # Create query for knowledge base
        query = f"Block with {block_features.get('tx_count', 0)} transactions, "
        query += f"average gas price {block_features.get('gas_price_mean', 0):.2e}"
        
        # Get prediction with explanation
        result = self.model.retrieve_and_generate(query, torch.FloatTensor(feature_vector))
        
        return result
    
    def save_model(self, path: str = "rag_model"):
        """Save the trained model"""
        Path(path).mkdir(exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        
        # Save knowledge base
        self.model.knowledge_base.save(f"{path}/knowledge_base")
        
        # Save config
        with open(f"{path}/config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        # Load model state
        self.model.load_state_dict(torch.load(f"{path}/model.pt"))
        
        # Load knowledge base
        self.model.knowledge_base.load(f"{path}/knowledge_base")
        
        logger.info(f"Model loaded from {path}")


def main():
    """Main function to demonstrate RAG system usage"""
    # Initialize system
    config = RAGConfig()
    rag_system = FrontRunningRAGSystem(config)
    
    # Prepare knowledge base
    rag_system.prepare_knowledge_base()
    
    # Train model
    rag_system.train_model()
    
    # Save trained model
    rag_system.save_model()
    
    # Example detection
    sample_features = {
        'gas_price_mean': 50e9,
        'gas_price_std': 10e9,
        'gas_price_min': 20e9,
        'gas_price_max': 200e9,
        'total_value': 100.5,
        'avg_value': 2.1,
        'value_std': 5.2,
        'total_gas_used': 12000000,
        'avg_gas_used': 250000,
        'tx_count': 48
    }
    
    result = rag_system.detect_front_running(sample_features)
    print("Detection Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()