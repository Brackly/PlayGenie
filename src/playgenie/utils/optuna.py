from typing import Union,Optional
import config
import optuna

def create_study(study_name:Union[str, None],
                      direction:str='minimize') -> optuna.study.Study:
    study_name = config.optuna_config.OPTUNA_STUDY_NAME if study_name is None else study_name
    study = optuna.create_study(study_name=study_name,
                                    direction=direction,
                                    storage=config.optuna_config.OPTUNA_STORAGE)
    return study


def load_study(study_name:Union[str, None]) -> optuna.study.Study:
    return optuna.load_study(study_name,storage=config.optuna_config.OPTUNA_STORAGE)