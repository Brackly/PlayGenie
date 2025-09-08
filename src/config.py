import os
import pandas as pd
import torch
from dotenv import load_dotenv
from dataclasses import dataclass
# Load environment variables from .env file
load_dotenv()

@dataclass
class WandbConfig:
    WANDB_ENTITY: str = os.getenv("WANDB_ENTITY")
    WANDB_PROJECT: str = os.getenv("WANDB_PROJECT")
    WANDB_API_KEY: str = os.getenv("WANDB_API_KEY")
    WANDB_HOST: str = os.getenv("WANDB_HOST")
    WANDB_RUN_NAME:str = pd.Timestamp.now().isoformat()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelConfig:
    INPUT_SIZE:str = 384
    CONTEXT_SIZE:int = 10

@dataclass
class TrainingConfig:
    EPOCHS: int = 100
    OPTUNA_EPOCHS:int = 20
    TRAIN_RATIO:float = 0.6
    VAL_RATIO:float = 0.2
    OPTUNA_TRIALS: int = 50
    HYPER_PARAM_DATA_RATIO: float = 0.5

@dataclass
class PathConfig:
    DATA_PATH = './data/spotify.safetensors'
    MODEL_SAVE_PATH = './data/model.pt'
    MODEL_NAME = 'PlayGenie'
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDINGS_FILE_PATH="./data/song_embeddings.parquet"
    OUTPUT_PATH = './data/'

@dataclass
class OptunaConfig:
    OPTUNA_STUDY_NAME = pd.Timestamp.today().isoformat()
    OPTUNA_STORAGE = "sqlite:///optuna.db"


# =============================================================
# Singleton instances
# =============================================================

paths = PathConfig()
wandb_config = WandbConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
optuna_config = OptunaConfig()