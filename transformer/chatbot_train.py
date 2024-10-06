from encoder import TransformerEncoder, TransformerEncoderLayer
from decoder import TransformerDecoder, TransformerDecoderLayer
from transformer import Transformer
from positionalEncoding import PositionalEncoding
from inputEmbedding import InputEmbedding
# from embedding import embedding_dim
import pandas as pd

import torch.nn as nn
import torch
import numpy as np

MAX_LENGTH = 30
embedding_dim = MAX_LENGTH + 2

# https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv
train_data = pd.read_csv('ChatBotData.csv')
rows = train_data.iterrows()
vocab_size = train_data['Q'].count()

# 입력 데이터
input_embedding = InputEmbedding(embedding_dim, vocab_size, 0.1)

# 위치 임베딩
positional_encoding = PositionalEncoding(embedding_dim, 0.1, 5000)

def train(model, loss_fn, optimizer):
  # input_texts = ["나는 최근 파리 여행을 다녀왔다."] * vocab_size
  for i, input_text in enumerate(rows):
    question = ' '.join(np.pad(input_text[1]['Q'].split(), (0, MAX_LENGTH - len(input_text[1]['Q'].split())), 'constant', constant_values='0'))
    answer = ' '.join(np.pad(input_text[1]['A'].split(), (0, MAX_LENGTH - len(input_text[1]['A'].split())), 'constant', constant_values='0'))
    question = 'START ' + question + ' END'
    answer = 'START ' + answer + ' END'

    Y = positional_encoding(
      input_embedding(answer)
    )

    pred = model(question)

    loss = loss_fn(pred, Y)
    loss.backward()
    # gradient vanishing or gradient exploding 방지를 위해 gradient clipping 적용
    # 안정적인 학습을을 도와줌
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
      print(f'[{str(i).ljust(5)} / {str(vocab_size).ljust(5)}] Loss:{loss.item()}', )


if __name__ == '__main__':
  # 인코더
  encoder = TransformerEncoder(
    TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.01),
    num_layers=6, 
  )
  
  # 디코더
  decoder = TransformerDecoder(
    TransformerDecoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.01),
    num_layers=6
  )

  # 트랜스포머
  transformerModel = Transformer(input_embedding, positional_encoding, encoder, decoder)

  # 손실 함수
  loss_fn = nn.CrossEntropyLoss()
  # 옵티마이저: 확률적 경사하강법
  optimizer = torch.optim.SGD(transformerModel.parameters(), lr=1e-4)
  epoch = 1

  print('transformer model summary')
  print(transformerModel)

  for i in range(epoch):
    print(f"Epoch {i+1}\n----------------------------")
    train(transformerModel, loss_fn, optimizer)

  torch.save(transformerModel.state_dict(), "model.pth")
  print('Done!')