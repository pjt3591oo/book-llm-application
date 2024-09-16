from transformers import AutoTokenizer

# tokenizer.json
# tokenizer_config.json
model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenized = tokenizer('토크나이저는 텍스트를 토큰 단위로 나눈다.')
'''
{
  'input_ids': [0, 9157, 7461, 2190, 2259, 8509, 2138, 1793, 2855, 5385, 2200, 20950, 18, 2], 
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

attention_mask는 0 또는 1이 될 수 있다.
  0: padding
  1: 토큰이 있는 위치
'''
print(tokenized)


'''
['[CLS]', '토크', '##나이', '##저', '##는', '텍스트', '##를', '토', '##큰', '단위', '##로', '나눈다', '.', '[SEP]']
'''
print(tokenizer.convert_ids_to_tokens(tokenized.input_ids))

'''
[CLS] 토크나이저는 텍스트를 토큰 단위로 나눈다. [SEP]
'''
print(tokenizer.decode(tokenized.input_ids))

'''
토크나이저는 텍스트를 토큰 단위로 나눈다.
'''
print(tokenizer.decode(tokenized.input_ids, skip_special_tokens=True))

