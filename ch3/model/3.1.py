# AutoModel: 모델의 바디를 불러오는 클래스
from transformers import AutoModel, AutoTokenizer

text = "What is Huggingface Transformers?"

# huggingface의 모델은 Body와 Head로 구분된다.

bert_model = AutoModel.from_pretrained("bert-base-uncased") # BERT 모델을 불러옵니다.
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # BERT의 토크나이저를 불러옵니다.
encoded_input = bert_tokenizer(text, return_tensors='pt') # 입력 텍스트를 토큰화하고 인코딩합니다.
output = bert_model(**encoded_input) # 인코딩된 입력을 BERT 모델에 넣어 출력을 계산합니다.
print(output)

gpt_model = AutoModel.from_pretrained("gpt2") # GPT 모델을 불러옵니다.
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2") # GPT의 토크나이저를 불러옵니다.
encoded_input = gpt_tokenizer(text, return_tensors='pt') # 입력 텍스트를 토큰화하고 인코딩합니다.
output = gpt_model(**encoded_input) # 인코딩된 입력을 GPT 모델에 넣어 출력을 계산합니다.
print(output)