import torch.nn as nn
import torch
import copy
from math import sqrt
import torch.nn.functional as F
from multiheadAttention import MultiheadAttention
from embedding import embedding_dim

# 피드 포워드 층
class PreLayerNormFeedForward(nn.Module):
  def __init(self, d_model, dim_feedforward, dropout):
    super().__init__()
    self.linear1 = nn.Linear(d_model, dim_feedforward) # 선형층 1
    self.linear2 = nn.Linear(dim_feedforward, d_model) # 선형층 2
    self.dropout1 = nn.Dropout(dropout) # 드롭아웃 층 1
    self.dropout2 = nn.Dropout(dropout) # 드롭아웃 층 2
    self.activation = nn.GELU() # 활성함수
    self.norm = nn.LayerNorm(d_model) # 층 정규화
  
  def forward(self, src):
    x = self.norm(src)
    x = x + self.linear2(self.dropout1(self.activation(self.linear1(x))))
    x = self.dropout2(x)
    
    return x
  
def compute_attention(querys, keys, values, is_causal=False):
	dim_k = querys.size(-1) # 16
	scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k) # (1, 5, 5)
	if is_causal:
		query_length = querys.size(2)
		key_length = keys.size(2)
		temp_mask = torch.ones(query_length, key_length, dtype=torch.bool).tril(diagonal=0)
		scores = scores.masked_fill(temp_mask == False, float("-inf"))
	weights = F.softmax(scores, dim=-1) # (1, 5, 5)
	return weights @ values # (1, 5, 16)

class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
    super().__init__()
    self.self_attn = MultiheadAttention(d_model, d_model, nhead)
    self.multihead_attn = MultiheadAttention(d_model, d_model, nhead)
    self.feed_forward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, tgt, encoder_output, is_causal=True):
    # 셀프 어텐션 연산
    x = self.norm1(tgt)
    x = x + self.dropout1(self.self_attn(x, x, x, is_causal=is_causal))
    # 크로스 어텐션 연산
    x = self.norm2(x)
    x = x + self.dropout2(self.multihead_attn(x, encoder_output, encoder_output))
    # 피드 포워드 연산
    x = self.feed_forward(x)
    return x
  

def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(nn.Module):
  def __init__(self, decoder_layer, num_layers):
    super().__init__()
    self.layers = get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers

  def forward(self, tgt, src):
    output = tgt
    for mod in self.layers:
        output = mod(tgt, src)
    return output