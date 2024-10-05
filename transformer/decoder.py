import torch.nn as nn
import torch
import copy
from multiheadAttention import MultiheadAttention
from feedForwardLayer import PreLayerNormFeedForward
from embedding import embedding_dim
  
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
    x = x + self.dropout1(
      self.self_attn(x, x, x, is_causal)
    )
    
    # 크로스 어텐션 연산
    x = self.norm2(x)
    x = x + self.dropout2(
      self.multihead_attn(x, encoder_output, encoder_output)
    )

    # 피드 포워드 연산
    x = self.feed_forward(x)
    return x
  
## nn.ModuleList를 이용하여 디코더 층을 여러개 쌓음
## nn.ModuleList를 이용해야 nn.ModuleList의 파라미터들이 학습이 됨
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
  
if __name__ == '__main__':
  # 디코더 층
  decoder_layer = TransformerDecoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
  # 디코더
  decoder = TransformerDecoder(decoder_layer, num_layers=6)
  # 입력 데이터
  tgt = torch.rand(10, 32, embedding_dim)
  src = torch.rand(10, 32, embedding_dim)

  # 디코더 연산
  output = decoder(tgt, src)
  
  print(output.size())