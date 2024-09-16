# BERT 토크나이저와 RoBERTa 토크나이저

from transformers import AutoTokenizer



bert_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
'''
{
  'input_ids': [
    [2, 1656, 1141, 3135, 6265, 3, 864, 1141, 3135, 6265, 3]
  ],
  'token_type_ids': [
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  ], 
  'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  ]
}
'''
print(bert_tokenizer([['첫 번째 문장', '두 번째 문장']]))



roberta_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
'''
{
  'input_ids': [
    [0, 1656, 1141, 3135, 6265, 2, 864, 1141, 3135, 6265, 2]
  ], 
  'token_type_ids': [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ], 
  'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  ]
}
'''
print(roberta_tokenizer([['첫 번째 문장', '두 번째 문장']]))



en_roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
'''
{
  'input_ids': [
    [0, 43998, 14292, 4958, 47672, 14292, 23133, 43998, 6248, 18537, 47672, 11582, 18537, 43998, 17772, 8210, 2, 2, 45209, 3602, 16948, 47672, 14292, 23133, 43998, 6248, 18537, 47672, 11582, 18537, 43998, 17772, 8210, 2]
  ], 
  'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  ]
}
'''
print(en_roberta_tokenizer([['첫 번째 문장', '두 번째 문장']]))

