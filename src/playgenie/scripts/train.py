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


def create_model(hyper_params:dict)-> torch.nn.Module:
    model = VAE(input_size=config.model_config.INPUT_SIZE,
                hidden_size=hyper_params.get("hidden_size"),
                latent_size=hyper_params.get("latent_size"),
                encoder_n_heads=hyper_params.get("encoder_n_heads"),
                decoder_n_la=hyper_params.get("decoder_n_la"))
    return model

def create_optimizer(model, hyper_params:dict)-> torch.optim.Adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.get("learning_rate"))
    return optimizer


def train_model(hyper_params:dict,
                hyperparameter_tuning:bool=False)-> float:

    if not hyperparameter_tuning:
        wandb_utils = WandbUtils()
        wandb_utils.initialize(
            run_name= config.wandb_config.wandb_run_name,
            run_config = {
                **hyper_params,
                'input_size':config.model_config.INPUT_SIZE,
                'context_size': config.model_config.CONTEXT_SIZE,
                'epochs': config.training_config.EPOCHS,
                'optuna_trials': config.optuna_config.OPTUNA_TRIALS,
            }
        )

    train_dataloader = get_dataloader(mode='train')
    validation_dataloader = get_dataloader(mode='val')

    model = create_model(hyper_params=hyper_params)
    optim = create_optimizer(model=model, hyper_params=hyper_params)

    best_loss = float('inf')
    epochs = config.training_config.EPOCHS if hyperparameter_tuning else config.optuna_config.OPTUNA_EPOCHS

    epoch_bar = tqdm(range(epochs),desc=f'Training model',leave=False)
    for epoch in epoch_bar:
        avg_loss = 0
        avg_val_loss = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.squeeze(dim=0), targets.squeeze(dim=0)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model = model.to(DEVICE)
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
                val_inputs, val_targets = val_inputs.squeeze(dim=0), val_targets.squeeze(dim=0)
                val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
                val_preds,val_mu,val_log_var = model(val_inputs)

                loss = generative_loss(preds=val_preds,targets=val_targets,mu= val_mu, log_var= val_log_var)
                avg_val_loss += loss.item()

        avg_val_loss /= len(validation_dataloader)

        if not hyperparameter_tuning:
            wandb_utils.log_metrics(metrics={
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch
                })

        epoch_bar.set_postfix(Epoch=(epoch+1),Train_Loss=avg_loss,Validation_loss=avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if not hyperparameter_tuning:
                torch.save(model.state_dict(), config.paths.MODEL_SAVE_PATH)
                wandb_utils.log_artifact(
                    name=config.paths.MODEL_NAME,
                    artifact_path=config.paths.MODEL_SAVE_PATH,
                    artifact_type='model',
                    aliases= [f'epoch_{epoch}'],
                    tags= [f'epoch_{epoch}']
                )

                if os.path.exists(config.paths.MODEL_SAVE_PATH):
                    os.remove(config.paths.MODEL_SAVE_PATH)
        if (epoch+1) % 100 == 0:
            tqdm.write(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    return best_loss

def objective(trial:optuna.Trial)-> float:

    hyper_params = {
        "hidden_size": trial.suggest_int("hidden_size", 10, 90),
        "latent_size": trial.suggest_int("latent_size", 2, 50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3),
        'encoder_n_heads': trial.suggest_categorical("encoder_n_heads", [2,4,6,8,12]),
        'decoder_n_la': trial.suggest_int("decoder_n_la", 2, 10),
        }
    return train_model(hyper_params=hyper_params,hyperparameter_tuning=True)
