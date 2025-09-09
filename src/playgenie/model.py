from typing import Tuple,List,Union

import math
import torch
import config

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = True):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # projectors
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=bias)

        # final output projection
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        b, t, d = x.shape
        # (b, t, d_model) -> (b, t, num_heads, d_k)
        q = self.q_proj(x).view(b, t, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # (b, h, t, d_k)
        k = self.k_proj(x).view(b, t, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(b, t, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # attention scores: (b, h, t, d_k) @ (b, h, d_k, t) -> (b, h, t, t)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # causal mask: allow attending to current and previous tokens only
        # mask shape should be (1, 1, t, t) so it broadcasts over batch and heads
        device = x.device
        mask = torch.tril(torch.ones((t, t), device=device)).unsqueeze(0).unsqueeze(0)  # (1,1,t,t)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (b, h, t, t)

        # attention output (b, h, t, d_k)
        out = torch.matmul(attn, v)

        # concat heads: (b, h, t, d_k) -> (b, t, h*d_k=d_model)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, d)

        # final linear projection
        out = self.out_proj(out)  # (b, t, d_model)
        return out


class AttentionBlock(torch.nn.Module):
    """
    A single attention block that uses true multi-head causal self-attention.
    This preserves the general structure you had: an "attention" block that
    maps (b, t, d_model) -> (b, t, d_model).
    """
    def __init__(self, d_model: int, num_heads: int):
        super(AttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        # optional small feed-forward layer inside block (like your original feed_forward)
        # keep it simple: project d_model -> d_model (like a per-position FF)
        self.ff = torch.nn.Linear(d_model, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        attn_out = self.mha(x)           # (b, t, d_model)
        out = self.relu(self.ff(attn_out))
        return out

class Encoder(torch.nn.Module):
    """
    Encoder now takes `n_heads` as the number of heads inside the single AttentionBlock.
    If you want stacked layers, pass n_layers and construct ModuleList accordingly.
    """
    def __init__(self, input_size:int, hidden_size:int, latent_size:int, n_heads:int):
        super(Encoder, self).__init__()

        # Use one attention block that does multi-head attention.
        # Keep interface similar: input_size == d_model
        self.attention_block = AttentionBlock(d_model=input_size, num_heads=n_heads)

        # After attention block we aggregate across time (mean pooling)
        # and then map to mean/logvar.
        # hidden_size is expected to match the attention output dimension or be different.
        # We'll assume we want to map from input_size (d_model) -> hidden_size first.
        self.to_hidden = torch.nn.Linear(input_size, hidden_size)
        self.output_mean = torch.nn.Linear(hidden_size, latent_size)
        self.output_logvar = torch.nn.Linear(hidden_size, latent_size)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: (batch, seq_len, input_size)
        returns:
          mean: (batch, latent_size)
          log_var: (batch, latent_size)
        """
        # single attention block
        out = self.attention_block(inputs)        # (b, t, input_size)
        out = self.relu(self.to_hidden(out))      # -> (b,t, hidden_size)
        mean = self.output_mean(out)              # (b,t, latent_size)
        log_var = self.output_logvar(out)         # (b,t, latent_size)
        return mean, log_var

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
    def __init__(self, input_size:int, hidden_size:int, latent_size:int, encoder_n_heads:int=2,decoder_n_la:int=2):
        super(VAE, self).__init__()
        '''
        input_size: size of input features
        hidden_size: size of hidden layers
        latent_size: size of latent layers
        n_la: number of layers
        '''
        self.latent_size = latent_size

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size, n_heads=encoder_n_heads)
        self.decoder = Decoder(latent_size=latent_size, hidden_size=hidden_size, output_size=input_size, n_la=decoder_n_la)

    def forward(self, inputs:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        mu,log_var = self.encoder(inputs)
        out = reparameterize(mu=mu,log_var=log_var)
        out = self.decoder(out)
        return out, mu, log_var

    @torch.no_grad()
    def generate(self,
                 playlist:Union[List[torch.Tensor],None]=None
                 ) -> List[torch.Tensor]:

        if playlist is None:
            noise = torch.randn(1, 2)
            first_song = self.decoder(noise).detach()
            return first_song[0]
        else:
            next_song, _, _ = self(torch.cat([song.view(1,-1) for song in playlist],dim=1).view(1, -1, config.model_config.INPUT_SIZE))
            return next_song[0, 0]
