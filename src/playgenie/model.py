from typing import Tuple

import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_size:int, hidden_size:int, latent_size,n_la:int=2):
        super(Encoder, self).__init__()

        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.mid_layers = torch.nn.Sequential()
        for  _ in range(n_la):
            self.mid_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.mid_layers.append(torch.nn.ReLU())

        self.output_layer = torch.nn.Linear(hidden_size, latent_size)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.relu(self.input_layer(inputs))
        out = self.mid_layers(out)
        mean:torch.Tensor = self.output_layer(out)
        log_var: torch.Tensor = self.output_layer(out)
        return mean,log_var

class Decoder(torch.nn.Module):
    def __init__(self, latent_size:int, hidden_size:int, output_size,n_la):
        super(Decoder, self).__init__()
        self.input_layer = torch.nn.Linear(latent_size, hidden_size)
        self.mid_layers = torch.nn.Sequential()
        for _ in range(n_la):
            self.mid_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.mid_layers.append(torch.nn.ReLU())

        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
    def forward(self, latent:torch.Tensor) -> torch.Tensor:
        out = self.relu(self.input_layer(latent))
        out = self.mid_layers(out)
        out = self.relu(self.output_layer(out))
        return out


def reparameterize(mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
    epsilon = torch.randn_like(log_var)
    return mu + log_var * epsilon


class VAE(torch.nn.Module):
    def __init__(self, input_size:int, hidden_size:int, latent_size:int, encoder_n_la:int=2,decoder_n_la:int=2):
        super(VAE, self).__init__()
        '''
        input_size: size of input features
        hidden_size: size of hidden layers
        latent_size: size of latent layers
        n_la: number of layers
        '''
        self.latent_size = latent_size

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size, n_la=encoder_n_la)
        self.decoder = Decoder(latent_size=latent_size, hidden_size=hidden_size, output_size=input_size, n_la=decoder_n_la)

    def forward(self, inputs:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        mu,log_var = self.encoder(inputs)
        out = reparameterize(mu=mu,log_var=log_var)
        out = self.decoder(out)
        return out, mu, log_var

    def generate(self, batch_size:int) -> torch.Tensor:
        noise  = torch.randn(batch_size, self.latent_size)
        return self.decoder(noise)

