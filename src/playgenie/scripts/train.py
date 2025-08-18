import os
import torch
from playgenie.model import VAE
from playgenie.loss import generative_loss
import config
from tqdm import tqdm
from playgenie.data.dataset import get_dataloader
import optuna
from playgenie.utils.wandb import WandbUtils

DEVICE = config.DEVICE
wandb_utils = WandbUtils()

def create_model(hyper_params:dict)-> torch.nn.Module:
    model = VAE(input_size=config.INPUT_DIM, hidden_size=hyper_params.get("HIDDEN_DIM"), latent_size=hyper_params.get("LATENT_DIM"))
    return model

def create_optimizer(model, hyper_params:dict)-> torch.optim.Adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.get("LEARNING_RATE"))
    return optimizer

def train_loop(
        hyper_params:dict)-> float:

    wandb_utils.initialize(
        run_name= config.WANDB_RUN_NAME,
        run_config = {
            **hyper_params,
            'input_dim': config.INPUT_DIM,
            'context_size': config.CONTEXT_SIZE,
            'epochs': config.EPOCHS,
            'optuna_trials': config.OPTUNA_TRIALS,
        }
    )

    train_dataloader = get_dataloader(path=config.DATA_PATH,mode='train',batch_size=64)
    validation_dataloader = get_dataloader(path=config.DATA_PATH,mode='validation',batch_size=64)

    model = create_model(hyper_params=hyper_params)
    optim = create_optimizer(model=model, hyper_params=hyper_params)

    best_loss = float('inf')

    epoch_bar = tqdm(range(config.EPOCHS),desc=f'Training model',leave=False)
    for epoch in epoch_bar:
        avg_loss = 0
        avg_val_loss = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds,mu,log_var = model(inputs)

            optim.zero_grad()
            loss = generative_loss(preds=preds,targets=targets,mu= mu, log_var= log_var)
            loss.backward()
            optim.step()

            avg_loss += loss.item()

        avg_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in validation_dataloader:
                val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
                val_preds,val_mu,val_log_var = model(val_inputs)

                loss = generative_loss(preds=val_preds,targets=val_targets,mu= val_mu, log_var= val_log_var)
                avg_val_loss += loss.item()

        avg_val_loss /= len(validation_dataloader)

        wandb_utils.log_metrics(metrics={
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'epoch': epoch
            })

        epoch_bar.set_postfix(Epoch=(epoch+1),Train_Loss=avg_loss,Validation_loss=avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            wandb_utils.log_artifact(
                name=config.MODEL_NAME,
                artifact_path=config.MODEL_SAVE_PATH,
                artifact_type='model',
                aliases= [f'epoch_{epoch}'],
                tags= [f'epoch_{epoch}']
            )

            if os.path.exists(config.MODEL_SAVE_PATH):
                os.remove(config.MODEL_SAVE_PATH)
        if (epoch+1) % 100 == 0:
            tqdm.write(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    return best_loss

def objective(trial:optuna.Trial)-> float:

    hyper_params = {
        "HIDDEN_DIM": trial.suggest_int("HIDDEN_DIM", 10, 90),
        "LATENT_DIM": trial.suggest_int("LATENT_DIM", 2, 50),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1e-4, 1e-3),
        }
    return train_loop(hyper_params=hyper_params)
