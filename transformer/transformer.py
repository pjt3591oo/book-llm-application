import torch.nn as nn

class Transformer(nn.Module):

  def __init__(self, input_embedding, positional_encoding, encoder, decoder):
    super(Transformer, self).__init__()
    self.input_embedding = input_embedding
    self.positional_encoding = positional_encoding
    self.encoder = encoder
    self.decoder = decoder

  def encode(self, x):
    out = self.encoder(x)
    return out

  def decode(self, z, c):
    out = self.decoder(z, c)
    return out

  def forward(self, x):
    # TODO: input_embedding, positional_encoding는 encoder와 decoder에서 따로 사용되도록 변경
    # 별도의 학습 가중치가 필요함
    x = self.input_embedding(x)
    x = self.positional_encoding(x)
    
    c = self.encode(x)
    y = self.decode(x, c)

    return y