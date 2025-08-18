import logging

from playgenie.scripts.train import objective
from playgenie.utils.optuna import create_load_study
import config
from playgenie.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting experiment..")
    study = create_load_study(study_name=config.OPTUNA_STUDY_NAME)
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS)

    best_optuna_trial = study.best_trial

    print(f"{best_optuna_trial=}")


if __name__ == "__main__":
    main()
