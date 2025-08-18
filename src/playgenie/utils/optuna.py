from typing import Union
import config
import optuna

def create_load_study(study_name:Union[str, None]) -> optuna.study.Study:
    study_name = config.OPTUNA_STUDY_NAME if study_name is None else study_name
    study = optuna.create_study(study_name=study_name)
    return study