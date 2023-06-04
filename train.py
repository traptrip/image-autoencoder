import hydra
from omegaconf import DictConfig

from src.training import Trainer


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    run()
