import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CustomPipeline:
  
  def __init__(self, model_id):
    self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model.eval()
  
  def __call__(self, texts):
    tokenized = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
      outputs = self.model(**tokenized)
      logits = outputs.logits
    
    probabilities = softmax(logits, dim=1)
    scores, labels = torch.max(probabilities, dim=1)
    labels_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]

    return [{"label": label, "score": score} for label, score in zip(labels_str, scores)]


model_id = 'klue/roberta-base'
custom_pipeline = CustomPipeline(model_id)
rst = custom_pipeline("컴퓨터")
print(rst)