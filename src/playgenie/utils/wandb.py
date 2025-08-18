from typing import List

import config
import wandb

class WandbUtils:
    def __init__(self):
        wandb.login(
            key=config.WANDB_API_KEY,
            host=config.WANDB_HOST,
            relogin=True

        )

    @staticmethod
    def initialize(run_config:dict,
                   run_name:str):
        wandb_run:wandb.run = wandb.init(
            project = config.WANDB_PROJECT,
            entity = config.WANDB_ENTITY,
            name = run_name,
            config=run_config)
        return wandb_run

    @staticmethod
    def log_artifact(artifact_path:str,
                     name:str,
                     artifact_type:str,
                     aliases:List[str],
                     tags:List[str]):
        wandb.log_artifact(
            artifact_or_path = artifact_path,
            name=name,
            type=artifact_type,
            aliases = aliases,
            tags = tags
        )

    @staticmethod
    def log_metrics(metrics:dict):
        wandb.log(data=metrics)
