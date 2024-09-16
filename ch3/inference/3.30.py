from transformers import pipeline

model_id = 'klue/roberta-base'

model_pipeline = pipeline("text-classification", model=model_id)

rst = model_pipeline("컴퓨터")
print(rst)


model_id = 'SamLowe/roberta-base-go_emotions'

model_pipeline = pipeline("text-classification", model=model_id)

rst = model_pipeline("today is payday!")
print(rst)