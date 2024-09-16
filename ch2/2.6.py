# 절대적 위치 인코딩
import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
from embedding import input_embeddings, embedding_dim

##### 쿼리, 키, 값, 벡터를 만드는 nn.Linear 층 #####
head_dim = 16

# 쿼리, 키, 값을계산하기 위한 변환
weight_q = nn.Linear(embedding_dim, head_dim)
weight_k = nn.Linear(embedding_dim, head_dim)
weight_v = nn.Linear(embedding_dim, head_dim)

print("weight_q:", weight_q)
print("weight_k:", weight_k)
print("weight_v:", weight_v)

querys = weight_q(input_embeddings)
keys = weight_k(input_embeddings)
values = weight_v(input_embeddings)

print("querys:", querys)
print("keys:", keys)
print("values:", values)

##### 쿼리, 키, 값, 벡터를 만드는 nn.Linear 층 #####


##### 스케일 점곱 방식의 어텐션 #####

def compute_attention(querys, keys, values, is_causal=False):
    dim_k = querys.size(-1)
    scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ values

after_attention_embeddings = compute_attention(querys, keys, values)
print("어텐션 적용 후 형태: ", after_attention_embeddings)
print("어텐션 적용 후 형태: ", after_attention_embeddings.shape)

##### 스케일 점곱 방식의 어텐션 #####