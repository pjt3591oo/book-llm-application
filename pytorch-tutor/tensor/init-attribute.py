import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(data)
print(x_data)
print(np_array)
print(x_np)


print(f"Shape of tensor: {x_data.shape}")
print(f"Datatype of tensor: {x_data.dtype}")
print(f"Device tensor is stored on: {x_data.device}")

print(f"Shape of tensor: {x_np.shape}")
print(f"Datatype of tensor: {x_np.dtype}")
print(f"Device tensor is stored on: {x_np.device}")

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")


x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

if torch.cuda.is_available():
  tensor = rand_tensor.to('cuda')
  print(tensor)

print('tensor operation')

print('================= index & slice =================')
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

print('================= concat =================')
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

print('================= arithmetic operations =================')
data = [[0.1, 0.2], [0.3, 0.4]]
tensor = torch.tensor(data)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z2)
print(z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
# add_, copy_, t_와 같은 연산은 도함수 계산에 문제가 발생할 수 있어 사용을 권장하지 않음
tensor.add_(5) # 모든 요소에 5를 더함 
# tensor.copy_(tensor) # tensor의 값을 복사
# tensor.t_() # tensor를 전치
print(tensor)

print('================= bridge with numpy =================')
print('tensor to numpy & numpy to tensor')

# tensor와 numpy 배열은 메모리 공간을 공유하기 때문에 하나를 변경하면 다른 하나도 변경됨

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 3, out=n)
print(f"t: {t}")
print(f"n: {n}")