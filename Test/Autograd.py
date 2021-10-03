import torch

# the most simple network
# requires_grad is property value of a tensor
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

loss.backward()
print(w.grad)
print(b.grad)

# Disabling Gradient Tracking
# 将神经网络中的某些参数标记为冻结参数。这是微调预训练网络的一个非常常见的场景
# 在仅进行前向传递时加快计算速度，因为对不跟踪梯度的张量进行计算会更有效。
# case 1
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
# case 2
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)