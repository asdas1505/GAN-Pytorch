import numpy as np
import pandas as pd 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import *
from model import *


def train_function(x):

    encoder.train()
    decoder.train()
    discriminator.train()

## Traning VAE i.e. encoder and decoder
    encoder.zero_grad()
    decoder.zero_grad()
    discriminator.zero_grad()
    
    x_recon = decoder(encoder(x))

    loss_vae = F.binary_cross_entropy(x_recon, x)
    loss_vae.backward()
    decoder_optim.step()
    encoder_optim.step()

## Training discriminator to distinguish samples from aggregated posterior and prior on the latent space 
    z_qz = encoder(x)
    z_pz = Variable(torch.randn(bs, latent_dim) * 5.)

    discriminator.zero_grad()

    dis_loss = -torch.mean(torch.log(discriminator(z_pz)) + torch.log(1-discriminator(z_qz)))
    dis_loss.backward()
    disc_optim.step()
    
    encoder.zero_grad()
    gen_encoder_loss = -torch.mean(torch.log(discriminator(encoder(x))))
    gen_encoder_loss.backward()
    disc_encoder_optim.step()
    
    return loss_vae.item(), dis_loss.item(), gen_encoder_loss.item()


bs = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


latent_dim = 8

### For MNIST dataset 

encoder = Encoder(inp = 784, out=latent_dim)
encoder = encoder.apply(init_weights)
decoder = Decoder(inp = latent_dim, out = 784)
decoder = decoder.apply(init_weights)
discriminator = Discriminator(inp=latent_dim)
discriminator = discriminator.apply(init_weights)



encoder_optim = optim.Adam(encoder.parameters(), lr=1e-4)
decoder_optim = optim.Adam(decoder.parameters(), lr=1e-4)
disc_encoder_optim = optim.Adam(encoder.parameters(), lr=6e-5)
disc_optim = optim.Adam(discriminator.parameters(), lr=6e-5)


### Training 

epochs = 30

discriminator_losses = []
vae_losses = []
gen_losses = []

for epoch in range(epochs):
    train_vae_loss = 0
    train_dis_loss = 0
    train_gen_loss = 0
    
    num_of_iter = 0
    for X_train, _ in train_loader:
        X_train = Variable(X_train.view(bs,-1))
        vae_l, dis_l, gen_l = train_function(X_train)
        train_vae_loss += vae_l
        train_dis_loss += dis_l
        train_gen_loss += gen_l
        num_of_iter += 1
        
    discriminator_losses.append(train_dis_loss/num_of_iter)
    vae_losses.append(train_vae_loss/num_of_iter)
    gen_losses.append(train_gen_loss/num_of_iter)

    reconstruct_images(X_train, epoch)
    
    print('Training Epoch: {}, VAE loss = {:.3f}, Discriminator Loss = {:.3f} and Generator Loss = {:.3f} '.format(epoch+1, train_vae_loss/num_of_iter, train_dis_loss/num_of_iter, train_gen_loss/num_of_iter))



training_plots(vae_losses)

images_path = 'samples/'
save_path = 'samples'

latent_space_visualization(test_loader)
make_gif(images_path, save_path)