import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
from embedding import input_embeddings, embedding_dim

norm = nn.LayerNorm(embedding_dim)
norm_x = norm(input_embeddings)

# 평균과 표준편차
print(norm_x.mean(dim=-1), norm_x.std(dim=-1).data)