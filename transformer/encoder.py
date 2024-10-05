import torch
import torch.nn as nn
import copy
from multiheadAttention import MultiheadAttention
from embedding import embedding_dim
from feedForwardLayer import PreLayerNormFeedForward

# 인코더 층
class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout):
    super().__init__()
    self.attn=MultiheadAttention(d_model, d_model, nhead) # 멀티헤드 어텐션 클래스
    self.norm1 = nn.LayerNorm(d_model) # 층 정규화 
    self.dropout1 = nn.Dropout(dropout) # 드롭아웃
    self.feed_forward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout) # 피드 포워드 층

  def forward(self, src):
    norm_x = self.norm1(src)
    attn_output = self.attn(norm_x, norm_x, norm_x)
    x = src + self.dropout1(attn_output) # 잔차연결

    # 피드 포워드
    x = self.feed_forward(x)
    return x
  

# 인코더 구현
## nn.ModuleList를 이용하여 인코더 층을 여러개 쌓음
## nn.ModuleList를 이용해야 nn.ModuleList의 파라미터들이 학습이 됨
def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
  def __init__(self, encoder_layer, num_layers):
    super().__init__()
    self.layers = get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = nn.LayerNorm(embedding_dim)

  def forward(self, src):
    output = src 
    
    for mod in self.layers:
      output = mod(output)
    
    return output
  
if __name__ == '__main__':
  # 인코더 층
  encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
  # 인코더
  encoder = TransformerEncoder(encoder_layer, num_layers=6)
  # 입력 데이터
  src = torch.rand(10, 32, embedding_dim)

  # 인코더 연산
  output = encoder(src)
  
  print(output.size())