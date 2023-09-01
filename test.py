import torch
import torch.nn as nn
import torch.optim as optim

# 创建随机输入和输出数据
input_data = torch.randn(32, 8, 4)
output_data = torch.randn(32, 8, 2)

# 输入张量的维度和输出张量的维度
input_dim = 4
output_dim = 2

# 创建线性模型
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to (batch_size * 8, 4) before applying the linear layer
        # x = x.view(-1, x.size(-1))  # Flatten the last dimension
        y = self.linear(x)
        # Reshape the output tensor to (batch_size, 8, 2)
        # y = y.view(x.size(0), x.size(1), -1)
        return y

# 创建模型实例
model = LinearModel(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(input_data)

    # 计算损失
    loss = criterion(outputs, output_data)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练过程中的损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 输出学习到的矩阵
for name, param in model.named_parameters():
    print(f'{name}:')
    print(param)