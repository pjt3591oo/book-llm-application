import torch.nn as nn
import copy
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