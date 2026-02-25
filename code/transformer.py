import torch
import torch.nn as nn
import math
from transformers import BertModel
from collections import Counter
import re
# Part 1: Data preparation

# Step 1: Buiuld a Vocabulary class to transfrom human language to numbers
class Vocabulary:
    def __init__(self, min_freq=2):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"([.,!?()])", r" \1 ", text) # Separate punctuation
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4 

        for sentence in sentence_list:
            for word in self.tokenize(str(sentence)):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        # Return index of word, or the <unk> index if not found
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]
    
    
# Step 2: Build a container that organizes the data for PyTorch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, vocab, max_len=300):
        self.df = dataframe
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        raw_label = self.df.iloc[idx]['label']
        
        # Convert to IDs
        numerical_text = self.vocab.numericalize(text)
        
        # For CrossEntropyLoss, labels must be integers (0 or 1)
        label = 1 if raw_label == 'CG' else 0
        
        # Add <sos> and <eos>
        ids = [self.vocab.stoi["<sos>"]] + numerical_text + [self.vocab.stoi["<eos>"]]
        
        # Padding & Truncating
        if len(ids) < self.max_len:
            ids += [self.vocab.stoi["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
            
        # Returning label as long (int64) instead of float
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# Part 2: Transformer architechture

# Step 1: The Input Layer
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        # d_model: The dimensionality of the vector for each word. 
        # If this is 512, every word in your review becomes a list of 512 numbers.
        # vocab_size: The total number of unique words in your dataset
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Scale the embeddings by sqrt(d_model) as per the original paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        # seq_len: The maximum length of a review (e.g., 500 words)
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add the fixed positional encoding to the word embeddings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
    
# Step 2: The Processor (Encoder)
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.h = h # h = number of heads
        self.d_k = d_model // h # d_k = dimension per head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self.w_q(q).view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

        # Dot-product attention
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        
        x = (self.dropout(attention_scores) @ value)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.norm1 = nn.LayerNorm(features)
        self.norm2 = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Attention + Residual Connection
        x2 = self.self_attention_block(x, x, x, mask)
        x = self.norm1(x + self.dropout(x2))
        # Feed Forward + Residual Connection
        x2 = self.feed_forward_block(x)
        x = self.norm2(x + self.dropout(x2))
        return x
    
# Step 3: The Classification Output
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        # For fake review detection, num_classes would be 2 (Real/Fake)
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # We usually take the first token's output or the average of the sequence
        return self.proj(x)
    
class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=512, n_layers=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, seq_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(
                d_model, 
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(n_layers)
        ])
        
        self.projection = ProjectionLayer(d_model, 2)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
            
        # POOLING: Instead of mean, take the first token ([sos])
        # x is (batch, seq_len, d_model) -> we take x[:, 0, :]
        # Result is (batch, d_model)
        summary_vector = x[:, 0, :]
        
        return self.projection(summary_vector)
    
    
# Part 3: Model training

# Step 1: Load and split the data
import pandas as pd
from sklearn.model_selection import train_test_split 
df = pd.read_csv('../data/data_for_transformer.csv')

df_train, df_temp = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

df_val, df_test = train_test_split(
    df_temp, 
    test_size=0.5, 
    random_state=42, 
    stratify=df_temp['label']
)

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# Step 2: Prepare the data loader
# Build vocab
vocab = Vocabulary(min_freq=2)
vocab.build_vocabulary(df_train['text'].tolist())
print(f"Vocabulary Size: {len(vocab)}")

# Create datasets 
train_dataset = Dataset(df_train, vocab, max_len=300)
val_dataset = Dataset(df_val, vocab, max_len=300)
test_dataset = Dataset(df_test, vocab, max_len=300)

# Create DataLoaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Step 3: Initialize the model, optimizer, loss function
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

import torch.optim as optim

# Initialize Model, Loss, and Optimizer
model = ReviewClassifier(vocab_size=len(vocab), seq_len=300).to(DEVICE)
criterion = nn.CrossEntropyLoss() # Good for 2-class classification
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Step 4: Train the model
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    pad_idx = vocab.stoi["<pad>"]
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0
        
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            
            # 1. Create the Padding Mask (batch, 1, 1, seq_len)
            # This tells Attention to ignore the <pad> tokens
            mask = (ids != pad_idx).unsqueeze(1).unsqueeze(2)
            
            # 2. Forward pass
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            
            # 3. Backward pass
            loss.backward()
            
            # 4. Gradient Clipping (Crucial for Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            total_train += labels.size(0)
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / total_train

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad(): # No gradients needed for validation
            for ids, labels in val_loader:
                ids, labels = ids.to(device), labels.to(device)
                mask = (ids != pad_idx).unsqueeze(1).unsqueeze(2)
                
                logits = model(ids, mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                total_val += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / total_val
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 30)

# Execute training
train_model(model, train_loader, val_loader, optimizer, criterion, DEVICE, epochs=10)