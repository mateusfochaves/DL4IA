''' Define the attention layers used in the transformer model
Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module 
    Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    Minimal modifications: head axis broadcasting at line 79, q, k, v args -> x at line 68, compute_values option.'''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, compute_values=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.compute_values = compute_values

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        if compute_values:
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x, mask=None):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)

        residual = x

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_k, n_head, d_k)
        if self.compute_values:
            v = self.w_vs(x).view(sz_b, len_v, n_head, d_v)
        else:
            v = x.view(sz_b, len_v, 1, -1).repeat(1, 1, n_head, 1)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1).unsqueeze(2)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
        

class LearnableQueryMultiHeadAttention(nn.Module):
    '''TODO: update the Multi-Head Attention module with a learnable query
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, compute_values=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.compute_values = compute_values

        # A single learnable query token.
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        if compute_values:
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_k, len_v = x.size(0), x.size(1), x.size(1)
        len_q = 1

        q_input = self.query.expand(sz_b, -1, -1)
        residual = q_input

        q = self.w_qs(q_input).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_k, n_head, d_k)
        if self.compute_values:
            v = self.w_vs(x).view(sz_b, len_v, n_head, d_v)
        else:
            v = x.view(sz_b, len_v, 1, -1).repeat(1, 1, n_head, 1)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # mask: (B, L) -> (B, n_head, 1, L)
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1).unsqueeze(2)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention 
    Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn
    

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module 
    Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x