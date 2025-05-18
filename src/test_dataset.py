import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import logging
log = logging.getLogger(__name__)


@hydra.main(version_base = None, config_path = "../configs", config_name = "test_dataset")
def main(config: DictConfig):
    dataset = instantiate(config.dataset)
    log.info("Fabric initialized")

    log.info(f"Dataset length: {len(dataset)}")
    idx, sample = dataset[0]
    log.info(f"Dataset sample shape: {sample.shape}")
    log.info(f"Dataset sample type: {type(sample)}")
    log.info(f"Dataset sample min max: {sample.min()} {sample.max()}")

if __name__ == "__main__":
    main()



