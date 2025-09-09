import torch
from playgenie.model import VAE


model  = VAE(input_size=384, hidden_size=10,latent_size=2,encoder_n_heads=2,decoder_n_la=2)
model.generate()