from transformers import AutoModelForSequenceClassification

# 분류 헤드가 포함되어 있다.
# 어떤 감성을 나타내는지 분류하는 모델을 불러옴
model_id = 'SamLowe/roberta-base-go_emotions' 
# 텍스트 시퀀스 분류를 위한 헤드가 포함된 모델을 불러올 때 사용하는 클래스
classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)
print(classification_model)