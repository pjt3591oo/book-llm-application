import torch
from torch import nn

from data_load import train_dataloader, test_dataloader
from model import model, device


def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # 순전파 단계
    # 예측 오류 계산
    pred = model(X) # model.forward(X)를 직접 호출하지 않는다.
    loss = loss_fn(pred, y)

    # 역전파 단계
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  
  test_loss /= num_batches
  correct /= size
 
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
  epochs = 5

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

  print("Done!")

  torch.save(model.state_dict(), "model.pth")
  print("Saved PyTorch Model State to model.pth")