import torch
import torch.nn as nn 
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout = 0.0):
        super(MultiHeeadAttention,self).__init__()
        assert d_model % num_heads == 0 , "d_model must be divisible by no. of heads"
        self.d_model = d_model
        self.num_heads = num_heads 
        self.d_k = d_model// num_heads
        self.attn_dropout =  nn.Dropout(dropout)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k= nn.Linear(d_model, d_model)
        self.W_v= nn.Linear(d_model, d_model)
        self.W_o= nn.Linear(d_model, d_model)
    def scaled_dot_product_attention(self, Q,K,V, mask = None):
        attn_scores= torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_k) 
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e4)
        attn_probs= torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        output= torch.matmul(attn_probs,V)
        return output
    def split_heads(self,x):
        batch_size, seq_length, d_model=x.size()
        return x.view(batch_size, seq_length, self.num_heads,self.d_k).transpose(1,2)
    def combine_heads(self,x):
        batch_size, _, seq_length , d_k= x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    def forward(self, Q, K, V, mask= None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model,d_ff):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1= nn.Linear(d_model,d_ff)
        self.fc2= nn.Linear(d_ff, d_model)
        self.relu = nn.GELU()
    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x,positions):
        return x + self.pe[0, positions.long()]
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads,d_ff, dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeeadAttention(d_model, num_heads,dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x,Tgt_mask):
        normed_x = self.norm1(x)
        attn_output = self.self_attn(normed_x, normed_x, normed_x, Tgt_mask)
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x
class GPT (nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_head,  num_layers, d_ff, max_seq_length, dropout ):
        super(GPT, self).__init__()
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding= PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_head, d_ff, dropout) for i in range(num_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size,bias=False)
        self.fc.weight = self.decoder_embedding.weight
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)                                  # NEW
        self.fc.weight = self.decoder_embedding.weight                  # NEW

    def _init_weights(self, module):                                    # NEW
        if isinstance(module, nn.Linear):                               # NEW
            nn.init.normal_(module.weight, mean=0.0, std=0.02)          # NEW
            if module.bias is not None:                                 # NEW
                nn.init.zeros_(module.bias)                             # NEW
        elif isinstance(module, nn.Embedding):                          # NEW
            nn.init.normal_(module.weight, mean=0.0, std=0.02)          # NEW

    def generate_mask(self, tgt, positions):
        seq_length = tgt.size(1)
        causal = torch.tril(torch.ones(seq_length, seq_length, device=tgt.device, dtype=torch.bool))
        doc_id = torch.cumsum(positions == 0, dim=1)
        same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)
        return (causal & same_doc).unsqueeze(1)
    
    def forward(self, tgt, positions):
        tgt_mask = self.generate_mask(tgt, positions)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt), positions))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)
        dec_output = self.norm_f(dec_output)
        output = self.fc(dec_output)
        return output