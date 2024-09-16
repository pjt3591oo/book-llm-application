# Trainer API를 이용하여 학습
import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from dataset import train_dataset, valid_dataset, test_dataset

from huggingface_hub import login

# trainer 사용 준비
model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(train_dataset.features['label'].names))

def tokenize_function(examples):
    return tokenizer(examples['title'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# trainer를 사용한 학습: 학습 인자와 평가 함수 정의
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # evaluation_strategy="epoch",
    learning_rate=5,
    push_to_hub=False
)

def compute_tetrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

# 학습 진행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_tetrics
)


# 학습 및 평가 
trainer.train()
trainer.evaluate(test_dataset)


# 학습한 모델 업로드
login(token='허깅페이스 토큰')
repo_id = f'pjt3591oo/roberta-base-klue-ynat-classification'
trainer.push_to_hub(repo_id)
