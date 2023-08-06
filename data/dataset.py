import sys

import torch
from torch.utils.data import Dataset, DataLoader
# from data.simdata import *
import numpy as np
from utils.data_helper import *
from scipy.stats import norm
from scipy import interpolate
import pandas as pd


class basedata():
    def __init__(self, n, n_feature=6) -> None:
        self.y = None
        self.t = None
        self.x = None
        self.num_data = n
        self.n_feature = n_feature
        self.load_data()
        self.true_pdf = self.get_correct_pdf()

    def load_data(self):
        # Load your preprocessed data from CSV
        data = pd.read_pickle('processed_data.pkl')
        data.fillna(0, inplace=True)
        columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        x = torch.tensor(data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
                                    'discipline', 'course_length']].values, dtype=torch.float32)
        t = torch.tensor(data['grade_reqs', 'nforum_posts', 'ndays_act'].values, dtype=torch.float32)
        y = torch.tensor(data['grade', 'explored', 'nevents', 'completed_%'].values, dtype=torch.float32)
        return x, t, y

    def get_dose(self, t):
        n = t.shape[0]
        x_tmp = torch.rand([10000, self.n_feature])
        dose = torch.zeros(n)
        for i in range(n):
            t_i = t[i]
            psi = self.get_outcome(x_tmp, t_i).mean()
            # psi /= n_test
            dose[i] = psi
        return dose

    def get_correct_conditional_desity(self, x, t):
        derivation_t = derivation_sigmoid(t).numpy()
        t = inverse_sigmoid(t)
        loc = self.set_pre_treatment(x)
        scale = 0.5
        pdf = norm.pdf(t, loc, scale) * derivation_t
        return pdf

    def get_correct_desity(self, t):
        x = torch.rand([10000, 6])
        cde = self.get_correct_conditional_desity(x, t)
        return torch.from_numpy(cde.mean(axis=1))

    def get_correct_pdf(self, x, t):
        derivation_t = derivation_sigmoid(t).numpy()
        t = inverse_sigmoid(t)
        loc = self.t
        scale = 0.5
        pdf = norm.pdf(t, loc, scale) * derivation_t
        return pdf

    def get_ideal_weights(self, x, t, power=0.5):
        t_ = t.reshape(-1, 1)
        conditional_de = self.get_correct_conditional_desity(x, t)
        des = torch.from_numpy(self.true_pdf(t_).squeeze())
        ideal_weights = des / conditional_de
        ideal_weights = torch.pow(ideal_weights, power)
        return ideal_weights
    # ... rest  ...


def get_iter(self, data, batch_size, shuffle=True, rw=False):
    dataset = basedata(data)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


# ---------------------------------------------------------


# data = pd.read_pickle('processed_data.pkl')
# data.fillna(0, inplace=True)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# num_rows, num_columns = data.shape
# print("Number of rows:", num_rows)
# print("Number of columns:", num_columns)
# print(data.head(47422))
# print(data[-1000:])  # 显示最后1000行数据

# 创建一个文件并写入内容
# with open('data_info.txt', 'w') as f:
#     original_stdout = sys.stdout  # 保存原始的标准输出对象
#     sys.stdout = f  # 将标准输出重定向到文件
#
#     # 打印所有数据
#     print(data)
#
#     sys.stdout = original_stdout  # 恢复原始的标准输出对象
#
# print("Data information saved to data_info.txt")

#
# columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
# for col in columns_to_convert:
#     data[col] = pd.to_numeric(data[col], errors='coerce')
# # print(data.dtypes)
#
# x = torch.tensor(data[['LoE_DI', 'age_DI', 'primary_reason', 'learner_type', 'expected_hours_week',
#                       'discipline', 'course_length']].values, dtype=torch.float32)
# t = torch.tensor(data[['grade_reqs', 'nforum_posts', 'ndays_act']].values, dtype=torch.float32)
# y = torch.tensor(data[['grade', 'explored', 'nevents', 'completed_%']].values, dtype=torch.float32)
#
# print("Loaded data:")
# print("x shape:", x.shape)
# print("t shape:", t.shape)
# print("y shape:", y.shape)

# LoE_DI'-1.0, 'age_DI'-1.0, 'primary_reason'3.0, 'learner_type'2.0, 'expected_hours_week2.0

# 打印数据+2 = 原始csv

# data_helper.py: 这个文件包含了一些数据处理的辅助函数，例如加载数据、保存数据、数据预处理等。
# 它定义了一些用于处理数据的函数，例如load_data、save_data、load_train、load_test等。
# 这些函数用于管理数据的读取和保存，以及对数据的预处理操作。data_helper.py 里的函数和逻辑有助于更好地组织和管理数据。
#
# dataset.py: 这是一个数据集定义的文件，可能在代码中没有给出，但在代码的其他部分可能会使用它。
# 在伪代码中，有一个 Dataset_from_simdata 类，它继承自 PyTorch 中的 Dataset 类。
# 这个类的作用是将原始的数据（在伪代码中的 data）转化为适合训练的数据集对象，以便在训练时可以方便地迭代和获取数据样本。
# 在实际代码中，这个文件可能会定义如何将原始数据转化为数据集对象，并实现 __len__ 和 __getitem__ 方法。
#
# 所以，data_helper.py 提供了对数据的处理和管理功能，而 dataset.py 则定义了数据集对象的构建方式，
# 两者共同协作以提供训练和测试数据的有效管理和使用。
