import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 检查GPU是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# define our neural network by subclassing nn.Module
# 继承nn.Module将自动跟踪模型对象中定义的所有字段，并使所有参数都
# 可以使用模型的parameters()或named_parameters()方法访问
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# create an instance of NeuralNetwork, and move it to the device
model = NeuralNetwork().to(device)
print(model)

# input a tensor and use softmax to get the predict result
X = torch.rand(4, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# check the parameters of the defined model using named_parameters() or parameters()
# 这里包括在模型中定义的所有字段
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")