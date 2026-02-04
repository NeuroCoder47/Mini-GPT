nimport os
import numpy as np
from model_GPT import *
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import math
import numpy as np
import random
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

out_dir = 'out'
data_dir = r'C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)'
batch_size = 4
block_size = 512
n_layer = 8
n_head = 16
n_embd = 512
dropout = 0.1
learning_rate = 3e-4
max_iters = 10000
weight_decay = 0.1
min_lr = 1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size   = 32768

class BinaryDataset(Dataset):
    def __init__(self, split, data_dir, block_size):
        path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        start = idx
        end = idx + self.block_size
        x_np = self.data[start : end].astype(np.int64)
        y_np = self.data[start+1 : end+1].astype(np.int64)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x, y

# ----- create datasets and loaders ONCE -----
train_dataset = BinaryDataset('train', data_dir, block_size)
val_dataset   = BinaryDataset('val',   data_dir, block_size)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,                         # shuffle per epoch
    pin_memory=(device == 'cuda'),
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device == 'cuda'),
    drop_last=True,
)

train_iter = iter(train_loader)
val_iter = iter(val_loader)

def get_batch(split):
    dataset = train_dataset if split == 'train' else val_dataset
    
    # Generate 'batch_size' number of random indices
    idxs = torch.randint(0, len(dataset), (batch_size,))
    
    # Fetch and stack the data
    x = torch.stack([dataset[i][0] for i in idxs])
    y = torch.stack([dataset[i][1] for i in idxs])
    
    return x.to(device), y.to(device)

criterion = nn.CrossEntropyLoss()
model = GPT(tgt_vocab_size=vocab_size, d_model=n_embd, num_head=n_head, num_layers=n_layer, d_ff=4*n_embd, max_seq_length=1024, dropout=dropout)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
model.to(device)

def evaluate(eval_iters=10): # Average over 10 batches
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch('val')
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


print("Training Stated......")
for step in range(max_iters):
    X, Y = get_batch('train')  
    logits = model(X) 
    batch_size, sequence_length, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size) 
    targets_flat = Y.view(-1)
    loss = criterion(logits_flat, targets_flat)
    optimizer.zero_grad()  
    loss.backward()  
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      
    optimizer.step()     

    if step % 100 == 0 and step > 0:  
        c = evaluate()

        print(f"Iteration {step}, Loss: {loss.item():.4f}, Val Loss {c:.4f}")
