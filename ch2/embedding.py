import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F

##### 토큰 아이디 생성 시작 #####

input_text = "나는 최근 파리 여행을 다녀왔다."
input_text_list = input_text.split()

str2idx = {word: idx for idx, word in enumerate(input_text_list)}
idx2str = {idx: word for idx, word in enumerate(input_text_list)}

input_ids = [str2idx[word] for word in input_text_list]

##### 토큰 아이디 생성 종료 #####

##### 절대적 위치 인코딩 #####
embedding_dim = 16
max_position = 12
embed_layer = nn.Embedding(len(str2idx), embedding_dim) # Embedding(5, 16)
position_embed_layer = nn.Embedding(max_position, embedding_dim) # Embedding(12, 16)

position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
position_encodings = position_embed_layer(position_ids)

token_embeddings = embed_layer(torch.tensor(input_ids))
token_embeddings = token_embeddings.unsqueeze(0) # shape: [1, 5, 16]

input_embeddings = token_embeddings + position_encodings