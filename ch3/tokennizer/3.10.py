from transformers import AutoTokenizer

# tokenizer.json
# tokenizer_config.json
model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenized0 = tokenizer(['첫 번째 문자', '두 번째 문장'])
'''
{
  'input_ids': [
    [0, 1656, 1141, 3135, 5703, 2], 
    [0, 864, 1141, 3135, 6265, 2]
  ], 
  'token_type_ids': [
    [0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0]
  ], 
  'attention_mask': [
    [1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1]
  ]
}
'''
print(tokenized0)

input_ids = tokenized0.input_ids
'''
['[CLS] 첫 번째 문자 [SEP]', '[CLS] 두 번째 문장 [SEP]']
'''
print(tokenizer.batch_decode(input_ids))



tokenized1 = tokenizer([['첫 번째 문자', '두 번째 문장']])
'''
{
  'input_ids': [
    [0, 1656, 1141, 3135, 5703, 2, 864, 1141, 3135, 6265, 2]
  ], 
  'token_type_ids': [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ], 
  'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  ]
}
'''
print(tokenized1)


input_ids = tokenized1.input_ids
'''
['[CLS] 첫 번째 문자 [SEP] 두 번째 문장 [SEP]']
'''
print(tokenizer.batch_decode(input_ids))