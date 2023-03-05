# _*_ coding:utf-8 _*_

from myutils import VAE, vae_loss
from my_dataset.my_dataset import load_data
from torch.utils.tensorboard import SummaryWriter
from math import inf
import torch
torch.manual_seed(0)

"""
Training VAE
"""

def train(model, loss_fn, lr, model_name, batchsize, device, epochs:int=100):
    writer = SummaryWriter(log_dir='experiments')
    model = model.to(device)
    # load dataset
    train_dataloader, val_dataloader = load_data(batch_size=batchsize)
    # define optimizer
    # optimizer = optimizer.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    expect_ls = inf
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for dna_data, rna_data, rppa_data, _ in train_dataloader:
            optimizer.zero_grad()
            if 'dna' in model_name.lower():
                x = dna_data.float().to(device)
                mean, log_var, z, recon_x = model(x)
            elif 'rna' in model_name.lower():
                x = rna_data.float().to(device)
                mean, log_var, z, recon_x = model(x)
            else:
                x = rppa_data.float().to(device)
                mean, log_var, z, recon_x = model(x)
            loss = loss_fn(recon_x, x, mean, log_var)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        avg_ls = loss_sum / len(train_dataloader)
        writer.add_scalar('train/' + model_name, avg_ls, epoch + 1)  
        print(f"Epoch {epoch + 1}: Average loss {avg_ls}")

        model.eval()
        loss_sum = 0.0
        with torch.no_grad():
            for dna_data, rna_data, rppa_data, _ in val_dataloader:
                if 'dna' in model_name.lower():
                    x = dna_data.float().to(device)
                    mean, log_var, z, recon_x = model(dna_data)
                elif 'rna' in model_name.lower():
                    x = rna_data.float().to(device)
                    mean, log_var, z, recon_x = model(rna_data)
                else:
                    x = rppa_data.float().to(device)
                    mean, log_var, z, recon_x = model(rppa_data)
                loss = loss_fn(recon_x, x, mean, log_var)
                loss_sum += loss.item()     
            avg_ls = loss_sum / len(train_dataloader)
            if avg_ls < expect_ls:
                expect_ls = avg_ls
                torch.save(model.state_dict(), 'models/'+model_name+'.pt')
        writer.add_scalar('validation/' + model_name, avg_ls, epoch + 1)    
        print(f"Epoch {epoch + 1}: Average loss {avg_ls}")                       
    writer.close()


if __name__ == "__main__":
    model1 = VAE(392799, [2048, 256])
    model1_name = 'DNA'
    model2 = VAE(18574, [1024, 512, 256, 128])
    model2_name = 'RNA'
    model3 = VAE(217, [128, 64])
    model3_name = 'RPPA'
    loss_fn1 = vae_loss
    loss_fn2 = vae_loss
    loss_fn3 = vae_loss
    device = 'cuda' if torch.cuda.is_available() else ' cpu'
    # device = 'cpu'
    # train(model1, loss_fn=loss_fn1, lr=1e-3, model_name=model1_name, 
    #       batchsize=2, device=device, epochs=100)
    train(model2, loss_fn=loss_fn2, lr=1e-3, model_name=model2_name, 
          batchsize=2, device=device, epochs=100)
    train(model3, loss_fn=loss_fn3, lr=1e-3, model_name=model3_name, 
          batchsize=2, device=device, epochs=100)    
    pass


