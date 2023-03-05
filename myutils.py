# _*_ coding: utf-8 _*_

import torch
from torch import nn 
import warnings
warnings.filterwarnings("ignore")


class VAE(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:list, dropout_p:float=0.5) -> None:
        super(VAE, self).__init__()
        hidden_len = len(hidden_dim)
        # Encoder
        if hidden_len <= 1:
            raise f'The length of hidden_dim at least is 2, but get {hidden_len}.'
        # elif hidden_len == 1:
        #     self.encoder = self.fc_layer(input_dim, hidden_dim[0])
        else:
            self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim[0]),
                    nn.BatchNorm1d(hidden_dim[0]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            for i in range(1, hidden_len-1):
                self.encoder.add_module(f'e_{i+1}_l',nn.Linear(hidden_dim[i-1], hidden_dim[i]))
                self.encoder.add_module(f'e_{i+1}_b',nn.BatchNorm1d(hidden_dim[i]))
                self.encoder.add_module(f'e_{i+1}_r',nn.ReLU())
                self.encoder.add_module(f'e_{i+1}_d',nn.Dropout(p=dropout_p))
            self.e_mean = nn.Sequential(
                nn.Linear(hidden_dim[-2], hidden_dim[-1]),
                nn.BatchNorm1d(hidden_dim[-1]))
            self.e_log_var = nn.Sequential(
                nn.Linear(hidden_dim[-2], hidden_dim[-1]),
                nn.BatchNorm1d(hidden_dim[-1]))
        # Decoder
        self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim[-1], hidden_dim[-2]),
                    nn.BatchNorm1d(hidden_dim[-2]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
        for i in range(2, hidden_len):
            self.decoder.add_module(f'd_{i}_l', nn.Linear(hidden_dim[-i], hidden_dim[-i-1]))
            self.decoder.add_module(f'd_{i}_b',nn.BatchNorm1d(hidden_dim[-i-1]))
            self.decoder.add_module(f'd_{i}_r',nn.ReLU())
            self.decoder.add_module(f'd_{i}_d',nn.Dropout(p=dropout_p))
        self.decoder.add_module('d_last_l',nn.Linear(hidden_dim[0], input_dim))
        self.decoder.add_module('d_last_b',nn.BatchNorm1d(input_dim))
        self.decoder.add_module('d_last_s',nn.Sigmoid())

    def encode(self, x):
        temp = self.encoder(x)
        mean = self.e_mean(temp)
        log_var = self.e_log_var(temp)
        return mean, log_var
    
    def reparameterization(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma
    
    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def forward(self, x):
        mean, log_var= self.encode(x)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decode(z)
        return mean, log_var, z, recon_x


def vae_loss(recon_x, x, mean, log_var):
    recon_ls = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl = - 0.5 * torch.sum(1 + log_var - mean.pow(2 - log_var.exp()))
    return recon_ls + kl

if __name__ == "__main__":
    pass