from typing import Tuple,List,Union

import numpy as np
import torch
import config

class CausalAttention(torch.nn.Module):
    def __init__(self,d_model,d_ff):
        super(CausalAttention, self).__init__()
        self.q = torch.nn.Linear(d_model, d_ff, bias=False)
        self.k = torch.nn.Linear(d_model, d_ff, bias=False)
        self.v = torch.nn.Linear(d_model, d_ff, bias=False)
        self.tril = torch.tril(torch.ones(d_ff, d_ff))
        self.feed_forward = torch.nn.Linear(d_ff, d_ff, bias=True)
        self.relu = torch.nn.ReLU()
    def forward(self,x:torch.Tensor):
        b,t,c = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        out = q @ k.transpose(-2,-1)
        out = out.masked_fill(self.tril[:t,:t]==0,float('-inf'))
        #
        out = torch.nn.functional.softmax(out, dim=-1)
        out = out @ v
        out = self.feed_forward(out)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, input_size:int, hidden_size:int, latent_size,n_la:int=2):
        super(Encoder, self).__init__()

        self.attention = CausalAttention(input_size, hidden_size)
        self.output_mean = torch.nn.Linear(hidden_size, latent_size)
        self.output_logvar = torch.nn.Linear(hidden_size, latent_size)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.relu(self.attention(inputs))
        mean:torch.Tensor = self.output_mean(out)
        log_var: torch.Tensor = self.output_logvar(out)
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
    var = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(var)
    return mu + var * epsilon


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

    @torch.no_grad()
    def generate(self,
                 batch_size:int=10,
                 playlist:Union[List[torch.Tensor],None]=None
                 ) -> List[torch.Tensor]:
        assert batch_size <= config.CONTEXT_SIZE , f"The batch size must not be greater than {config.CONTEXT_SIZE}"

        playlist = [] if playlist is None else playlist

        for song_idx in range(batch_size):
            if len(playlist) == 0:
                noise = torch.randn(1, 2)
                first_song = self.decoder(noise).detach()
                playlist.append(first_song[0])
            else:
                next_song, _, _ = self(torch.cat(playlist, dim=0).view(1, -1, config.INPUT_DIM))
                playlist.append(next_song[0, 0])
        return [song.detach().cpu().numpy() for song in playlist]
