import os

import torch

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 384
HIDDEN_DIM = 10
LATENT_DIM = 2