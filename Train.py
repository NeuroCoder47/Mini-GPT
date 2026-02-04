import os
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
import time
from torch.amp import autocast, GradScaler
from datatrove.utils.dataset import DatatroveFolderDataset

out_dir = 'out'
train_data_dir = r'C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\output_train'
val_data_dir = r'C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\output_test' 

batch_size = 8
block_size = 512
n_layer = 4
n_head = 16
n_embd = 512
dropout = 0.1
learning_rate = 3e-4
max_iters = 10000
weight_decay = 0.1
learning_rate = 6e-4      
min_lr = 6e-5             
lr_decay_iters = max_iters  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size   = 32768
max_batch_size = 32
warmup_iters = 50
token_size = 2
num_workers = 2


def get_batch(split, train_iter, val_iter):

    iterator = train_iter if split == 'train' else val_iter
    
    batch = next(iterator)
    input_ids = batch['input_ids']  
    
    x = input_ids[:, :-1] 
    y = input_ids[:, 1:]   
    if device == 'cuda':
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    
    return x, y


def evaluate(model, val_loader, criterion, vocab_size, device, eval_iters=4):
    model.eval()
    losses = []
    eval_iter = iter(val_loader)
    
    with torch.no_grad():
        for i in range(eval_iters):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(val_loader)
                batch = next(eval_iter)
            
            input_ids = batch['input_ids']
            x = input_ids[:, :-1].to(device, non_blocking=True)
            y = input_ids[:, 1:].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)


def lr_scheduling(learning_rate, min_lr, warmup_iters, it, lr_decay_iters):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def estimate_mfu(num_params, batch_size, block_size, grad_accum_steps, dt, device_type='cuda'):
    if device_type != 'cuda':
        return -1.0  
    flops_per_token = 6 * num_params
    tokens_per_iter = batch_size * block_size * grad_accum_steps
    flops_per_iter = flops_per_token * tokens_per_iter
    flops_achieved = flops_per_iter / dt
    flops_promised = 9.0e12 
    mfu = flops_achieved / flops_promised
    return mfu


if __name__ == '__main__':
    train_dataset = DatatroveFolderDataset(
        data_folder=train_data_dir,
        seq_len=block_size,
        filename_pattern="*.ds",
        recursive=True,
        token_size=token_size,
        shuffle=False,  
        return_positions=False,
    )


    val_dataset = DatatroveFolderDataset(
        data_folder=val_data_dir,
        seq_len=block_size,
        filename_pattern="*.ds",
        recursive=True,
        token_size=token_size,
        shuffle=False,
        return_positions=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )




    train_iter = iter(cycle(train_loader))
    val_iter = iter(cycle(val_loader))

    criterion = nn.CrossEntropyLoss()
    model = GPT(tgt_vocab_size=vocab_size, d_model=n_embd, num_head=n_head, num_layers=n_layer, d_ff=4*n_embd, max_seq_length=512, dropout=dropout)
    #model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,fused=True)
    scaler = torch.amp.GradScaler()
    print ("Using Mixed precision now")
    model.to(device)

    grad_accm_steps = max_batch_size // batch_size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    running_mfu = -1.0

    X, Y = get_batch('train', train_iter, val_iter) 

    print("Training Stated......")
    for step in range(max_iters):
        t0 = time.time()
        lr = lr_scheduling(learning_rate, min_lr, warmup_iters, step, lr_decay_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad()  
        for micro_step in range(grad_accm_steps):
            with autocast(device_type='cuda',dtype=torch.float16):
                logits = model(X) 
                batch_size_actual, sequence_length, vocab_size_actual = logits.shape        
                logits_flat = logits.view(-1, vocab_size_actual) 
                targets_flat = Y.view(-1)
                loss = criterion(logits_flat, targets_flat)/grad_accm_steps
            X, Y = get_batch('train', train_iter, val_iter) 
            scaler.scale(loss).backward() 
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update() 
        t1 = time.time()
        dt = t1 - t0 

        if step >= 5: 
            mfu = estimate_mfu(num_params, batch_size, block_size, grad_accm_steps, dt, device)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        if step % 10 == 0 and step > 0:
            print(f"[Step {step:5d}] Loss: {loss.item():.4f}, LR: {lr:.6f}, MFU: {running_mfu*100:.2f}%", flush=True)

        if step % 100 == 0 and step > 0:  
            c = evaluate(model, val_loader, criterion, vocab_size, device)
            print(f"Iteration {step}, Loss: {loss.item():.4f}, Val Loss {c:.4f}, LR: {lr:.6f}, MFU: {running_mfu*100:.2f}%")



