import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from collections import Counter
import re
from sklearn.model_selection import train_test_split
df = pd.read_csv('../data/data_for_lstm.csv')


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
    
    
vocab = Vocabulary(min_freq=2)
vocab.build_vocabulary(df_train['text'].tolist())
print(f"Vocabulary Size: {len(vocab)}")


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
        label = 1.0 if raw_label == 'CG' else 0.0
        
        # Add <sos> and <eos>
        ids = [self.vocab.stoi["<sos>"]] + numerical_text + [self.vocab.stoi["<eos>"]]
        
        # Padding & Truncating
        if len(ids) < self.max_len:
            ids += [self.vocab.stoi["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
            
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)
    
    

# Create datasets 
train_dataset = Dataset(df_train, vocab, max_len=300)
val_dataset = Dataset(df_val, vocab, max_len=300)
test_dataset = Dataset(df_test, vocab, max_len=300)

# Create DataLoaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        
        # Embedding Layer: Turns word IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM Layer: The layer that remembers word order
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Fully Connected Layer: Maps the LSTM output to our binary labels
        self.linear = nn.Linear(hidden_dim, output_dim)
        
        # Activation: Converts output to a probability between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, max_len]
        
        embedded = self.embedding(x) 
        # embedded shape: [batch_size, max_len, embed_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # hidden shape: [1, batch_size, hidden_dim]
        
        out = self.linear(hidden[-1])
        return self.sigmoid(out)
    
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = LSTM(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=1).to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)



epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        predictions = (outputs >= 0.5).float()
        
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        
    avg_loss = train_loss / len(train_loader)
    avg_acc = (train_correct / train_total) * 100
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

    model.eval()  
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():  
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            preds = (outputs >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_acc = (val_correct / val_total) * 100
    print(f"VAL Results | Loss: {val_loss/len(val_loader):.4f} | Acc: {avg_val_acc:.2f}%")
    
    
# Hyperparameter tuning
hidden_dims = [64, 128, 256]
learning_rates = [0.001, 0.0001]
best_overall_val_acc = 0
best_params = {}
epochs = 12

for hidden_dim in hidden_dims:
    for lr in learning_rates:
        print(f"\n--- Testing: Hidden Dim: {hidden_dim}, LR: {lr} ---")
        
        model = LSTM(vocab_size=len(vocab), embed_dim=100, hidden_dim=hidden_dim, output_dim=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                predictions = (outputs >= 0.5).float()
                
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
            avg_loss = train_loss / len(train_loader)
            avg_acc = (train_correct / train_total) * 100
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

            model.eval() # Set to evaluation mode
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad(): # Disable gradient calculation
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    # Accuracy calculation
                    preds = (outputs >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_acc = (val_correct / val_total) * 100
            print(f"VAL Results | Loss: {val_loss/len(val_loader):.4f} | Acc: {avg_val_acc:.2f}%")

            if avg_val_acc > best_overall_val_acc:
                best_overall_val_acc = avg_val_acc
                best_params = {'hidden_dim': hidden_dim, 'lr': lr}
                torch.save(model.state_dict(), 'best_grid_model.pth')
                print(f"New Global Best! Acc: {best_overall_val_acc:.2f}%")

print("Best parmas:", best_params)
print(f"Best results: {best_overall_val_acc:.2f}%")


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


best_h_dim = 256 
best_lr = 0.001

best_model = LSTM(vocab_size=len(vocab), embed_dim=100, hidden_dim=best_h_dim, output_dim=1).to(device)
best_model.load_state_dict(torch.load('best_grid_model.pth'))
best_model.eval() 

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        
        outputs = best_model(inputs).squeeze()
        preds = (outputs >= 0.5).float().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

import numpy as np
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 0 = Real (OR), 1 = Fake (CG)
print("--- Final Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=['Real (OR)', 'Fake (CG)']))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Final Confusion Matrix')
plt.show()