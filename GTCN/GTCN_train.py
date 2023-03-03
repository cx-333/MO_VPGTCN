# coding: utf-8


from cProfile import label
from pickletools import optimize
from time import process_time_ns
from tkinter import Menu
from GTNet.utils import norm_adj, data_split, clean_A, remove_edge_pts, separate
from GTNet.GNN_models import GAT, APPNP, DAGNN, GTCN2, TreeLSTM, GTAN2, GCN, GTAN, GTCN, SimpleGCN, SimpleGAT
import torch
import torch.nn.functional as F 
from torch import nn, Tensor
import numpy as np
import sys
sys.path.append('my_dataset/')
from my_dataset import *
sys.path.append('data_fusion/PFA/')
from PFA_main import PFA_main
import matplotlib.pyplot as plt
import argparse         # 调参
sys.path.append('data_reduce_dimension/')
from VAE.VAE import *
import math
from torch.nn.parameter import Parameter
import warnings
warnings.filterwarnings("ignore")


class GraphConvolution(nn.Module):
    def __init__(self, infeas, outfeas, bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = infeas
        self.out_features = outfeas
        self.weight = Parameter(torch.FloatTensor(infeas, outfeas))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outfeas))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
        '''
        for name, param in GraphConvolution.named_parameters(self):
            if 'weight' in name:
                #torch.nn.init.constant_(param, val=0.1)
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)
        '''

    def forward(self, x, adj):
        x1 = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        #self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        # F.elu() = {x x > 0; a(e^x - 1) x <= 0} ReLU的改进型
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)

        x = self.fc(x)

        return x           

def get_GTCN_g(A:Tensor, device) -> dict:
    A = A.to_sparse()
    A = clean_A(A)
    g = {}
    A1, A2 = separate(A, norm_type=1)
    A1 = A1.to(device)
    A2 = A2.to(device)
    g['edge_index'] = A1._indices()
    g['edge_weight1'], g['edge_weight2'] = A1._values(), A2
    return g

def get_adj(x):
    size = len(x)
    adj = torch.eye(size, size)
    i = 0
    while(i < size):
        j = i
        while(j < size):
            if(x[i]==x[j]):
                adj[i, j] = 1
                adj[j, i] = 1
            j += 1
        i += 1
    return adj

def corr_adj(x):
    output = torch.corrcoef(x)
    output[output > 0.7] = 1
    output[output <= 0.7] = 0
    return output


def parse_set() -> None:
    args = argparse.ArgumentParser()
    args.add_argument('--input_dim', type=int, default=128, help='input dimesion of model')
    args.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of model')
    args.add_argument('--output_dim', type=int, default=33, help='output dimension of model')

    result = args.parse_args()
    return result

class VAE_classifier(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, output_dim:int=2, class_dim:int=33) -> None:
        super(VAE_classifier, self).__init__()
        # 编码器
        self.encode = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(latent_dim, output_dim)
        self.log_var = nn.Linear(latent_dim, output_dim)
        # 解码器
        self.decode = nn.Sequential(
            nn.Linear(output_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()
        )
        # 分类器
        self.classifier = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(output_dim, class_dim),
            nn.Softmax(dim=1)
        )

    def encoder(self, x:Tensor) -> Tensor:
        temp = self.encode(x)
        mean = self.mean(temp)
        log_var = self.log_var(temp)
        return mean, log_var

    def reparameterization(self, mean:Tensor, log_var:Tensor) -> Tensor:
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        return mean + sigma * epsilon

    def decoder(self, z:Tensor) -> Tensor:
        recon_x = self.decode(z)
        return recon_x

    def forward(self, x:Tensor) -> Tensor:
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decoder(z)
        y_pre = self.classifier(mean)
        return mean, log_var, recon_x, z, y_pre

def train(device, epochs, lr, bs, model1, model2, model3, model, loss_fn):

    # 加载数据集
    train_data, val_data = load_data(batch_size=bs)

    # 加载模型
    model1 = model1.to(device)   # dna
    model2 = model2.to(device)   # rna
    model3 = model3.to(device)   # rppa
    # model = GTCN(n_in, n_hid, n_out, dropout, dropout2, hop)   # fusion data, concatenate data
    model = model.to(device)

    # 损失函数 loss_fn
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 记录参数
    train_acc, val_acc = [], []
    train_ls, val_ls = [], []

    for epoch in range(epochs):
        print('-'*50, 'Epoch: ', epoch+1, '-'*50)

        model.train()
        train_acc_sum, train_ls_sum = 0.0, 0.0
        for x1, x2, x3, y in train_data:
            x1 = x1.to(torch.float32).to(device)
            x2 = x2.to(torch.float32).to(device)
            x3 = x3.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)

            mean1, log_var1, recon_x1, z1, y_pre1 = model1(x1)   # dna
            mean2, log_var2, recon_x2, z2, y_pre2 = model2(x2)   # rna
            mean3, log_var3, recon_x3, z3, y_pre3 = model3(x3)   # rppa

            mean = torch.cat([mean1, mean2, mean3], axis=1)      # used for inputs to GTCN
            adj = PFA_main(mean1, mean2, mean3)
            # mean = mean.to(torch.float32).to(device)
            # adj = get_adj(y)
            adj = corr_adj(mean)
            # print(adj)
            adj = adj.to(device)
            g = get_GTCN_g(adj, device)

            pre_y = model(mean, g)

            optimizer.zero_grad()
            loss = loss_fn(pre_y, y)
            train_ls_sum += loss.cpu().item()
            acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
            train_acc_sum += acc_temp.cpu().item()
            loss.backward()
            optimizer.step()
            # print(acc_temp.cpu().item())
        train_acc.append((train_acc_sum/len(train_data))*100)
        train_ls.append(train_ls_sum/len(train_data))
        print("GTCN Average train loss : {:.4f} , Average train accuracy : {:.4f} "
              .format(train_ls_sum/len(train_data), (train_acc_sum/len(train_data))*100))

        model.eval()
        val_acc_sum, val_ls_sum = 0.0, 0.0
        with torch.no_grad():
            for x1, x2, x3, y in val_data:
                x1 = x1.to(torch.float32).to(device)
                x2 = x2.to(torch.float32).to(device)
                x3 = x3.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device)

                mean1, log_var1, recon_x1, z1, y_pre1 = model1(x1)   # dna
                mean2, log_var2, recon_x2, z2, y_pre2 = model2(x2)   # rna
                mean3, log_var3, recon_x3, z3, y_pre3 = model3(x3)   # rppa

                mean = torch.cat([mean1, mean2, mean3], axis=1)      # used for inputs to GTCN
                adj = PFA_main(mean1, mean2, mean3)
                # mean = mean.to(torch.float32).to(device)
                # adj = get_adj(y)
                # adj = corr_adj(mean)
                adj = adj.to(device)
                g = get_GTCN_g(adj, device)
                pre_y = model(mean, g)
                loss = loss_fn(pre_y, y)
                val_ls_sum += loss.cpu().item()
                acc_temp = torch.sum(torch.argmax(pre_y, axis=1) == y) / len(pre_y)
                val_acc_sum += acc_temp.cpu().item()
        val_acc.append((val_acc_sum/len(val_data))*100)
        val_ls.append(val_ls_sum/len(val_data))
        print("GTCN Average val loss : {:.4f} , Average val accuracy : {:.4f} "
              .format(val_ls_sum/len(val_data), (val_acc_sum/len(val_data))*100))       
        with open(r'C:\cx\paper\code\classification\GTCN\GTCN_acc.txt', 'r+', encoding='utf-8') as f:
            f.write('epoch: {}, val loss: {:.4f}, val acc: {:.4f}\n'.format(epoch+1, val_ls_sum/len(val_data),
                        (val_acc_sum/len(val_data))*100))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.plot(list(range(len(train_ls))), train_ls,'-', label='train loss')
    plt.plot(list(range(len(val_ls))), val_ls, '--', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('GTCN Loss Curve')
    plt.legend()
    # plt.savefig('data_reduce_dimension/VAE/result/scatter_tr.png')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(train_acc))), train_acc,'-', label='train accuracy')
    plt.plot(list(range(len(val_acc))), val_acc, '--', label='val accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('GTCN Accuracy Curve')
    plt.legend()
    plt.show() 


def class_loss(pre_y, y):
    # 分类损失函数（交叉熵）
    # print(torch.squeeze(pre_y).shape)
    return F.cross_entropy(torch.squeeze(pre_y), y.long(), reduction='sum')


if __name__ == "__main__":
    model1 = torch.load(r'C:\cx\paper\code\data_reduce_dimension\omics1.pkl')
    model2 = torch.load(r'C:\cx\paper\code\data_reduce_dimension\omics2.pkl')
    model1 = model1.module
    model3 = torch.load(r'C:\cx\paper\code\data_reduce_dimension\omics3.pkl')
    # print(model1, '\n', model2, '\n', model3)
    model = GTCN(420, 128, 33, 0.5, 0.25, 10)
    # model = GTCN(64, 50, 33, 0.5, 0.25, 10)
    device = torch.device('cuda')
    loss_fn = class_loss
    lr = 1e-3
    epochs = 100
    bs = 100
    train(device, epochs, lr, bs, model1, model2, model3, model, loss_fn)



