import torch
from torch.utils.data import DataLoader
from data.dataset import basedata  # 根据您的数据集文件路径和名称进行修改

class MyDataLoader:
    def __init__(self, n_samples, batch_size):
        self.n_samples = n_samples
        self.batch_size = batch_size

    def load_data(self):
        data = basedata(n=self.n_samples)  # 加载数据
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)  # 修改这里

        for batch in data_loader:
            x_batch, t_batch, y_batch = batch
            print("x_batch:", x_batch)
            print("t_batch:", t_batch)
            print("y_batch:", y_batch)

if __name__ == "__main__":
    n_samples = 1000
    batch_size = 32

    data_loader = MyDataLoader(n_samples, batch_size)
    data_loader.load_data()
