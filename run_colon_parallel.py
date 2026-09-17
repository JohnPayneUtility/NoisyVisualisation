# IMPORTS
import hydra
from omegaconf import DictConfig

from noisyvis.experiments.lon_runner import run_colon_experiment
from noisyvis.experiments.tracking import set_lon_module_tracking_uri

# MLflow defaults: established at import, before Hydra runs main (plan Stage 7, R24)
set_lon_module_tracking_uri()


@hydra.main(version_base=None, config_path="configs", config_name="test_lon_kp")
def main(cfg: DictConfig):
    run_colon_experiment(cfg)


if __name__ == "__main__":
    main()
