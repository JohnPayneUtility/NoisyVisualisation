# IMPORTS
from pathlib import Path

import hydra
from omegaconf import DictConfig

from noisyvis.experiments.config.cli import run_with_nested_config_name
from noisyvis.experiments.runner import run_so_experiment
from noisyvis.experiments.tracking import set_so_mo_tracking_uri

# explicit mlflow path: established at import, before Hydra runs main (plan Stage 7, R24)
set_so_mo_tracking_uri()


@hydra.main(version_base=None, config_path="configs", config_name="test1_kp_1p1")
def main(cfg: DictConfig):
    run_so_experiment(cfg)

if __name__ == '__main__':
    run_with_nested_config_name(main, Path(__file__).resolve().parent / "configs")
