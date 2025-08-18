import os
import pandas as pd
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_HOST = os.getenv("WANDB_HOST")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 384
CONTEXT_SIZE = 10
EPOCHS = 100
OPTUNA_TRIALS = 100

DATA_PATH = './data/spotify.safetensors'
MODEL_SAVE_PATH = './data/model.pt'
MODEL_NAME = 'PlayGenie'

OPTUNA_STUDY_NAME = pd.Timestamp.today().isoformat()
OPTUNA_STORAGE = "sqlite:///optuna.db"
WANDB_RUN_NAME = pd.Timestamp.now().isoformat()
