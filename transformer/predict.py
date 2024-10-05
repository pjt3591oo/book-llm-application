import torch
from encoder import TransformerEncoder, TransformerEncoderLayer
from decoder import TransformerDecoder, TransformerDecoderLayer
from inputEmbedding import InputEmbedding
from positionalEncoding import PositionalEncoding
from transformer import Transformer

embedding_dim = 32

device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)

if __name__ == "__main__":
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

  transformer = Transformer(input_embedding, positional_encoding, encoder, decoder)
  transformer.load_state_dict(torch.load("model.pth"))

  x = '나는 최근 파리 여행을 다녀왔다.'
  pred = transformer(x).to(device)

  print(pred)