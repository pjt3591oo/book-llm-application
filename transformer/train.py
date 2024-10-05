from encoder import TransformerEncoder, TransformerEncoderLayer
from decoder import TransformerDecoder, TransformerDecoderLayer
from transformer import Transformer
from positionalEncoding import PositionalEncoding
from inputEmbedding import InputEmbedding
# from embedding import embedding_dim

import torch.nn as nn
import torch


embedding_dim = 32

Y = torch.rand(5, 5, embedding_dim)

def train(model, loss_fn, optimizer):
  # TODO: dataset load
  vocab_size = 1000
  input_texts = ["나는 최근 파리 여행을 다녀왔다."] * vocab_size

  for i, input_text in enumerate(input_texts):
    pred = model(input_text)

    loss = loss_fn(pred, Y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
      print(f'[{str(i).ljust(5)} / {str(vocab_size).ljust(5)}] Loss:{loss.item()}', )


if __name__ == '__main__':
  # 인코더
  encoder = TransformerEncoder(
    TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.1),
    num_layers=6, 
  )
  
  # 디코더
  decoder = TransformerDecoder(
    TransformerDecoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.1),
    num_layers=6
  )

  # 입력 데이터
  input_embedding = InputEmbedding(embedding_dim, 10, 0.1)
  # 위치 임베딩
  positional_encoding = PositionalEncoding(embedding_dim, 0.1, 5000)

  # 트랜스포머
  transformerModel = Transformer(input_embedding, positional_encoding, encoder, decoder)

  # 손실 함수
  loss_fn = nn.CrossEntropyLoss()
  # 옵티마이저: 확률적 경사하강법
  optimizer = torch.optim.SGD(transformerModel.parameters(), lr=1e-3)
  epoch = 3

  for i in range(epoch):
    print(f"Epoch {i+1}\n----------------------------")
    train(transformerModel, loss_fn, optimizer)

  torch.save(transformerModel.state_dict(), "model.pth")
  print('Done!')