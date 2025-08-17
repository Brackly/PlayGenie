from playgenie.scripts.train import objective
from playgenie.utils.optuna import create_load_study
import config

def main(study_name:str=None):

    study = create_load_study(study_name=study_name)
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS)

    best_optuna_trial = study.best_trial

    print(f"{best_optuna_trial=}")


if __name__ == "__main__":
    main()
