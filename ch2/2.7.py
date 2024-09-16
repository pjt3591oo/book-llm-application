import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

from embedding import input_embeddings, embedding_dim

##### 쿼리, 키, 값, 벡터를 만드는 nn.Linear 층 #####
def compute_attention(querys, keys, values, is_causal=False):
    dim_k = querys.size(-1)
    scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ values

class AttentionHead(nn.Module):
    def __init__(self, token_embed_dim, head_dim, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.weight_q = nn.Linear(token_embed_dim, head_dim)
        self.weight_k = nn.Linear(token_embed_dim, head_dim)
        self.weight_v = nn.Linear(token_embed_dim, head_dim)
    
    def forward(self, querys, keys, values):
        return compute_attention(
            self.weight_q(querys),
            self.weight_k(keys),
            self.weight_v(values),
            is_causal=self.is_causal
        )


attention_head = AttentionHead(embedding_dim, embedding_dim)
after_attention_embeddings = attention_head(input_embeddings, input_embeddings, input_embeddings)

print("attention_head:", attention_head)
print("after_attention_embeddings:", after_attention_embeddings)
print(after_attention_embeddings.shape) # [1, 5, 16]