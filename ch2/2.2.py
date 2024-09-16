# 토큰 아이디에서 벡터로 변환
import torch.nn as nn
import torch

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

embedding_dim = 16
embed_layer = nn.Embedding(len(str2idx), embedding_dim)
print("embed_layer:", embed_layer)

input_embeddings = embed_layer(torch.tensor(input_ids))
print("input_embeddings:", input_embeddings) # shape: [5, 16]
input_embeddings = input_embeddings.unsqueeze(0) # shape: [1, 5, 16]
print("input_embeddings:", input_embeddings)

print(input_embeddings.shape) # [1, 5, 16]