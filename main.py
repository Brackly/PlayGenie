from typing import Literal, Optional
import argparse

from playgenie.scripts.train import objective,train_model
from playgenie.utils.optuna import create_study,load_study
import config
from playgenie.utils.logging import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser(
                    prog='Play Genie',
                    description='VAE model training')

def main():
    args = parser.parse_args()

    mode = args.mode
    study_name = args.study_name

    logger.info(f"Starting {mode}..")

    if mode=='hyperparam_optimization':
        study = create_study(study_name=study_name)
        study.optimize(objective, n_trials=config.training_config.OPTUNA_TRIALS)

        best_optuna_trial = study.best_trial

        print(f"{best_optuna_trial=}")
    else:
        if study_name is not None:
            study = load_study(study_name=study_name)
            best_params = study.best_trial.params
            train_model(hyper_params=best_params, hyperparameter_tuning=False)
        else:
            raise Exception("Study name must be provided to load the best parameters from optuna")



if __name__ == "__main__":
    parser.add_argument('--mode', help='whether model training or hyperparameter tuning',choices=['hyperparam_optimization', 'model_training'],required=True)
    parser.add_argument('--study_name', help='Optuna study name', default=None)
    try:
        main()
    except Exception as e:
        raise e
