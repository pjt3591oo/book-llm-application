from transformers import AutoModel


# config.json에서 model_type을 확인하여 해당 모델을 불러옴
model_id = 'klue/roberta-base'
model = AutoModel.from_pretrained(model_id)
print(model)