''' Transformer model 
Credits: https://github.com/jadore801120/attention-is-all-you-need-pytorch/
'''
import torch
import torch.nn as nn

from models.transformer.layers import PositionalEncoding,\
    Temporal_Aggregator, EmbeddingLayer, SpectralIndicesLayer

from models.transformer.attention import MultiHeadAttention,\
    LearnableQueryMultiHeadAttention, PositionwiseFeedForward


class Transformer(nn.Module):
    ''' A sequence to embedding model with attention mechanism. Major modifications made from 
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py'''

    def __init__(
            self, 
            n_channels, 
            n_pixels,
            d_model=512, 
            d_inner=2048,
            n_layers=6, 
            n_head=8, 
            d_k=64, 
            d_v=64, 
            dropout=0.1, 
            pad_value=0.,
            scale_emb_or_prj='prj', 
            n_position=365, 
            T=1000, 
            return_attns=False, 
            learnable_query=True, 
            spectral_indices_embedding=False,
            channels={}, 
            compute_values=True):

        super().__init__()

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']

        self.scale_emb = scale_emb_or_prj == 'emb'
        self.scale_proj = scale_emb_or_prj == 'prj'
        self.d_model = d_model
        self.n_head = n_head
        self.pad_value = pad_value
        self.return_attns = return_attns
        self.learnable_query = learnable_query
        
        if spectral_indices_embedding:
            self.embedding = SpectralIndicesLayer(d_model, **channels)
        else:
            self.embedding = EmbeddingLayer(n_channels, n_pixels, d_model)

        if learnable_query:
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, 
                             dropout=dropout, learnable_query=True, compute_values=compute_values)
                             ])
            self.temporal_aggregator = Temporal_Aggregator(mode='identity')
        else:
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, learnable_query=False)
                for _ in range(n_layers)])
            self.temporal_aggregator = Temporal_Aggregator(mode='mean')
        
        self.position_enc = PositionalEncoding(d_model, n_position=n_position, T=T)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
       
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def encoder(self, x, doys, mask):
        enc_slf_attn_list = []

        # -- Forward
        if self.scale_emb:
            enc_output = x * self.d_model ** 0.5
        else:
            enc_output = x

        device = x.device
        pos_embedding = self.position_enc(doys).to(device)
        enc_output = self.dropout(enc_output + pos_embedding)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask)
            enc_slf_attn_list += [enc_slf_attn] if self.return_attns else []

        if self.return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
       

    def forward(self, data, doys=None):
        '''TODO: implement the forward pass.
        '''
        if doys is None:
            doys = torch.zeros((data.shape[0], data.shape[1]))

        # Temporal mask (True for padded time steps)
        if data.dim() == 4:
            temporal_mask = (data == self.pad_value).all(dim=(2, 3))
        elif data.dim() == 3:
            temporal_mask = (data == self.pad_value).all(dim=-1)
        else:
            raise ValueError(f"Unexpected input shape: {tuple(data.shape)}")

        embeddings = self.embedding(data)
        if self.scale_proj:
            embeddings = embeddings * (self.d_model ** -0.5)
        
        if self.return_attns:
            enc_output, attns = self.encoder(embeddings, doys, temporal_mask)
        else:
            enc_output = self.encoder(embeddings, doys, temporal_mask)
            attns = None

        enc_output = self.temporal_aggregator(enc_output, temporal_mask)
        if enc_output.dim() == 3 and enc_output.shape[1] == 1:
            enc_output = enc_output[:, 0]

        return enc_output, attns
    

class EncoderLayer(nn.Module):
    ''' Compose with two layers
    Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Layers.py 
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, learnable_query=False, compute_values=True):
        super(EncoderLayer, self).__init__()
        if learnable_query:
            self.slf_attn = LearnableQueryMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, compute_values=compute_values)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn