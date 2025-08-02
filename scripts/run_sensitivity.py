# Add the parent directory to the Python path (because we are executing hydra from within runs)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.utils import seed_everything, data2pickle


# Registering the config path with Hydra - no hardcoded config_name
@hydra.main(config_path="../config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main function for running sensitivity analysis with sweep configurations.
    This script is designed to work with Hydra sweep configurations without conflicts.

    Args:
        cfg (DictConfig): Configuration object containing all parameters and sub-configurations.

    Returns:
        None: This function does not return any value.
    """

    ##############################
    # Preliminaries
    ##############################

    # directory for saving models and results
    results_save_dir = Path(cfg.save_dir)
    print(f"Results will be saved to: {results_save_dir}")

    # Set the random seed for reproducibility
    seed_everything(cfg.seed)

    ##############################
    # Object Instantiation
    ##############################

    # print out the full config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    if "seed" in cfg.objective.init:
        cfg.objective.init.seed = cfg.seed

    objective = hydra.utils.instantiate(cfg.objective.init)

    if "lb" in cfg.objective and "ub" in cfg.objective is not None:
        objective.bounds[0, :].fill_(cfg.objective.lb)
        objective.bounds[1, :].fill_(cfg.objective.ub)

    # initialize optimization algorithm
    opt_alg = hydra.utils.instantiate(
        cfg.optimizer.init,
        objective_function=objective,
        verbose=(cfg.verbose_level > 0),  # Convert verbose_level to boolean verbose
    )

    # Generate random x0 within the bounds
    lower_bound = objective.bounds[0]  # Extract lower bounds
    upper_bound = objective.bounds[1]  # Extract upper bounds

    x0 = lower_bound + torch.rand(opt_alg.dim) * (upper_bound - lower_bound)
    x0 = x0.reshape(1, -1)  # need explicit n x dim
    print(f"Initial point: {x0}")

    # Optimization
    res = opt_alg.minimize(
        n_initial=cfg.objective.n_initial,
        max_evals=cfg.objective.max_evals,
        x0=x0,
    )

    ##############################
    # Saving Results
    ##############################

    # create dir if not exists
    results_save_dir.mkdir(parents=True, exist_ok=True)

    # save results
    data2pickle(res, results_save_dir / "results.pkl")
    print(f"Results saved to {results_save_dir / 'results.pkl'}")

    # save yaml config
    with open(results_save_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    main() 