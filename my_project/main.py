import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

from my_project.train import train
from my_project.config import MyProjectConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=MyProjectConfig)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: MyProjectConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
