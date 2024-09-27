import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
from embedding import input_embeddings, embedding_dim

##### 쿼리, 키, 값, 벡터를 만드는 nn.Linear 층 #####
head_dim = 4
embedding_dim = 16

# 쿼리, 키, 값을계산하기 위한 변환
weight_q = nn.Linear(embedding_dim, head_dim)
querys = weight_q(input_embeddings)

print("weight_q:", querys)
print("weight_q:", querys.transpose(0, 1))