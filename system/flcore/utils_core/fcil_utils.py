import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import random


def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class Proxy_Data():
    def __init__(self, test_transform=None):
        super(Proxy_Data, self).__init__()
        self.test_transform = test_transform
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label,labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, new_set, new_set_label):
        datas, labels = [], []
        self.TestData, self.TestLabels = [], []
        if len(new_set) != 0 and len(new_set_label) != 0:
            datas = [exemplar for exemplar in new_set]
            for i in range(len(new_set)):
                length = len(datas[i])
                labels.append(np.full((length), new_set_label[i]))

        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        return img, target

    def __getitem__(self, index):
        if self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TestData != []:
            return self.TestData.shape[0]