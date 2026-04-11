from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    head_dim = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, 3 * head_dim)

def kqv(x, linear):
    B, N, D = x.size()
    qkv = linear(x) # (B, N, head_dim) @ (head_dim, 3 * head_dim) -> (B, N, 3 * head_dim)
    k, q, v = torch.chunk(qkv, chunks=3, dim=-1)
    return k, q, v

def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    A = b @ a.transpose(-2, -1) / math.sqrt(D1) # (b x n x head_dim) @ (b x head_dim x n) -> (b x n x n)
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):
    mask = torch.tril(torch.ones(max_context_len, max_context_len)) # torch.tril creates a lower triangular matrix (1s below diagonal, 0s above)
    mask = mask.view(1, max_context_len, max_context_len) # Reshape to add a leading dimension so it broadcasts across batches/heads correctly
    return mask

def self_attention(v, A, mask = None):
    if mask is not None: 
        # We slice the mask to N x N in case the sequence is shorter than max_context_len
        N = A.size(-1) 
        A = A.masked_fill(mask[:, :N, :N] == 0, float("-inf"))
        
    attention_weights = F.softmax(A, dim=-1) # (B, N, N)
    sa = attention_weights @ v # (B, N, N) @ (B, N, head_dim) -> (B, N, head_dim)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    head_outputs = []
    
    for kqv_matrix in kqv_matrices: # Run attention for each head individually
        head_sa = self_attention_layer(x, kqv_matrix, mask) # (B, N, head_dim), head_dim = D / n_heads
        head_outputs.append(head_sa)
        
    sa = torch.cat(head_outputs, dim=-1) # (B, N, D)
    
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask) # (B, N, D)
        sa = self.output_projection(sa)
        return sa
