import torch
from torch.utils.data import DataLoader
from data.dataset import basedata  # 根据您的数据集文件路径和名称进行修改

data = basedata(n=1000)  # 根据需要设置数据样本数
batch_size = 32

data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

for batch in data_loader:
    x_batch, t_batch, y_batch = batch
    print("x_batch:", x_batch)
    print("t_batch:", t_batch)
    print("y_batch:", y_batch)
    # 通过模型传递数据进行测试等操作
