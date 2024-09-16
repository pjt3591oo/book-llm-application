# Trainer API를 사용하지 않고 직접 학습
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from dataset import train_dataset, valid_dataset, test_dataset
import numpy as np
from huggingface_hub import login


# 학습을 위한 모델과 토크나이저 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "klue/roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(train_dataset.features['label'].names))
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.to(device)

def tokenize_function(example):
  return tokenizer(example["title"], padding="max_length", truncation=True)


# 학습을 위한 데이터 준비
# 전처리
def make_dataloader(dataset, batch_size, shuffle=True):
  # 데이터셋에 토큰화 수행
  dataset = dataset.map(tokenize_function, batched=True).with_format('torch')
  # 컬럼 이름 변경
  dataset = dataset.rename_column("label", "labels")
  # 불필요한 컬럼 제거
  dataset = dataset.remove_columns(column_names=["title"])
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
valid_loader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
test_loader = make_dataloader(test_dataset, batch_size=8, shuffle=False)


# 학습을 위한 함수 정의
def train_epoch(model, data_loader, optimizer):
  model.train()
  total_loss = 0

  for batch in tqdm(data_loader):
    optimizer.zero_grad()
    # 모델에 입력할 토큰 아이디
    input_ids = batch['input_ids'].to(device) 
    # 모델에 입력할 어텐션 마스트
    attention_mask = batch['attention_mask'].to(device) 
    # 모델에 입력할 레이블
    labels = batch['labels'].to(device) 
    # 모델계산
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels) 
    # 손실
    loss = outputs.loss 
    # 역전파
    loss.backward() 
    # 모델 업데이트
    optimizer.step() 
    total_loss += loss.item() 
    
    # # NOTE: 학습 시간이 너무 오래 걸려 중간에 중단
    # break
  
  avg_loss = total_loss / len(data_loader)
  
  return avg_loss


# 평가를 위한 함수 정의
def evaluate(model, data_loader):
  model.eval()
  total_loss = 0
  predictions = []
  true_labels = []

  with torch.no_grad():
    for batch in tqdm(data_loader):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      total_loss += loss.item()
      logits = outputs.logits
      preds = logits.argmax(dim=-1)
      predictions.extend(preds.cpu().numpy())
      true_labels.extend(labels.cpu().numpy())
  
  avg_loss = total_loss / len(data_loader)
  accuracy = np.mean(np.array(predictions) == np.array(true_labels))
  
  return avg_loss, accuracy



# 학습 수행
num_epochs = 1
# lr: 학습률
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
  print(f"epoch: {epoch+1} / {num_epochs}")
  train_loss = train_epoch(model, train_loader, optimizer)
  print(f"train_loss: {train_loss}")
  valid_loss, valid_acc = evaluate(model, valid_loader)
  print(f"valid_loss: {valid_loss}, valid_acc: {valid_acc}")

# 테스트 수행
test_loss, test_acc = evaluate(model, test_loader)
print(f"test_loss: {test_loss}, test_acc: {test_acc}")

# 학습한 모델 허깅페이스 업로드
login(token='허깅페이스 토큰')
repo_id = f'pjt3591oo/roberta-base-klue-ynat-classification'
model.push_to_hub(repo_id)