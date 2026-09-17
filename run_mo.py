# IMPORTS
import hydra
from omegaconf import DictConfig

from noisyvis.experiments.runner import run_mo_experiment
from noisyvis.experiments.tracking import set_so_mo_tracking_uri

# explicit mlflow path: established at import, before Hydra runs main (plan Stage 7, R24)
set_so_mo_tracking_uri()


@hydra.main(version_base=None, config_path="configs", config_name="test1_kp_1p1")
def main(cfg: DictConfig):
    run_mo_experiment(cfg)

if __name__ == '__main__':
    import sys
    from pathlib import Path

    configs_root = Path(__file__).resolve().parent / "configs"
    symlink = None

    for i, arg in enumerate(sys.argv):
        if arg.startswith("--config-name=") or arg.startswith("--config-name"):
            val = arg.split("=", 1)[1] if "=" in arg else sys.argv[i + 1]
            if "/" in val:
                flat = val.replace("/", "__")
                symlink = configs_root / f"{flat}.yaml"
                symlink.symlink_to((configs_root / f"{val}.yaml").resolve())
                if "=" in arg:
                    sys.argv[i] = f"--config-name={flat}"
                else:
                    sys.argv[i + 1] = flat
            break

    try:
        main()
    finally:
        if symlink and symlink.exists():
            symlink.unlink()