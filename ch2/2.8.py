import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
from embedding import input_embeddings, embedding_dim

##### 절대적 위치 인코딩 #####

##### 쿼리, 키, 값, 벡터를 만드는 nn.Linear 층 #####
def compute_attention(querys, keys, values, is_causal=False):
    dim_k = querys.size(-1)
    scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ values

class MultiheadAttention(nn.Module):
    def __init__(self, token_embed_dim, d_model, n_head, is_causal=False):
        super().__init__()
        self.n_head = n_head
        self.is_causal = is_causal
        self.weight_q = nn.Linear(token_embed_dim, d_model)
        self.weight_k = nn.Linear(token_embed_dim, d_model)
        self.weight_v = nn.Linear(token_embed_dim, d_model)
        self.concat_linear = nn.Linear(d_model, d_model)
    
    def forward(self, querys, keys, values):
        B, T, C = querys.size()
        querys = self.weight_q(querys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        keys = self.weight_k(keys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        values = self.weight_v(values).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attention = compute_attention(
            querys,
            keys,
            values,
            is_causal=self.is_causal
        )

        output = attention.transpose(1, 2).contiguous().view(B, T, C)
        output = self.concat_linear(output)
        return output

n_head = 4
mh_attention = MultiheadAttention(embedding_dim, embedding_dim, n_head)
after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)
print(after_attention_embeddings.shape) # [1, 5, 16]
