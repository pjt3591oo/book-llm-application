import torch.nn as nn
import torch

class InputEmbedding(nn.Module):
  def __init__(self, dim_model, vocab_size, dropout_p):
    super().__init__()

    self.dropout = nn.Dropout(dropout_p)
    self.token_embedding = nn.Embedding(vocab_size, dim_model)
    self.layer_norm = nn.LayerNorm(dim_model)

    self.str2idx = {}
    self.idx2str = {}

  def forward(self, input_text):
    input_ids = self.tokenized(input_text)
    embedding = self.token_embedding(torch.tensor(input_ids))
    
    return self.dropout(self.layer_norm(embedding))
    
  def tokenized(self, input_text):
    input_text_list = input_text.split()

    for idx, word in enumerate(input_text_list):
      if word not in self.str2idx:
        self.str2idx[word] = idx
        self.idx2str[idx] = word
    

    input_ids = [self.str2idx[word] for word in input_text_list]
    return input_ids