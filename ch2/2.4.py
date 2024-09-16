# 절대적 위치 인코딩
import torch.nn as nn
import torch

##### 토큰 아이디 생성 시작 #####

input_text = "나는 최근 파리 여행을 다녀왔다."
input_text_list = input_text.split()

print("input_text_list:", input_text_list)
print("")

str2idx = {word: idx for idx, word in enumerate(input_text_list)}
idx2str = {idx: word for idx, word in enumerate(input_text_list)}
print("str2idx:", str2idx)
print("idx2str:", idx2str)

input_ids = [str2idx[word] for word in input_text_list]
print("input_ids:", input_ids)

##### 토큰 아이디 생성 종료 #####

##### 절대적 위치 인코딩 #####
embedding_dim = 16
max_position = 12
embed_layer = nn.Embedding(len(str2idx), embedding_dim) # Embedding(5, 16)
position_embed_layer = nn.Embedding(max_position, embedding_dim) # Embedding(12, 16)
print("embed_layer:", embed_layer)
print("position_embed_layer:", position_embed_layer)

position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
position_encodings = position_embed_layer(position_ids)
print("position_ids:", position_ids)
print("position_encodings:", position_encodings)

token_embeddings = embed_layer(torch.tensor(input_ids))
print("token_embeddings:", token_embeddings) # shape: [5, 16]
token_embeddings = token_embeddings.unsqueeze(0) # shape: [1, 5, 16]
print("token_embeddings:", token_embeddings)

input_embeddings = token_embeddings + position_encodings
print("input_embeddings:", input_embeddings)

print(input_embeddings.shape) # [1, 5, 16]

##### 절대적 위치 인코딩 #####

##### 쿼리, 키, 값, 벡터를 만드는 nn.Linear 층
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