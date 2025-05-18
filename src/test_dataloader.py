import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.fabric import Fabric
import time

from utils.data_utils import InfiniteSampler

import logging
log = logging.getLogger(__name__)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_dataloader(config, fabric: Fabric):
    dataset = instantiate(config.dataset)
    sampler = InfiniteSampler(dataset, shuffle=True, seed=config.exp.seed, max_len=config.exp.max_iters)
    dataloader = instantiate(config.dataloader)(dataset=dataset, sampler=sampler)
    dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)
    return dataloader


@hydra.main(version_base = None, config_path = "../configs", config_name = "test_dataloader")
def main(config: DictConfig):
    torch.multiprocessing.set_start_method('spawn')

    fabric: Fabric = get_fabric(config)
    log.info("Fabric initialized")

    dataloader = get_dataloader(config, fabric)
    log.info("Dataloader initialized")

    for i, (idx, data) in enumerate(dataloader):
        if i == 0:
            log.info("Starting to iterate over dataloader")
            start_time = time.time()
            log.info(f"Data shape: {data.shape}")
            log.info(f"Data type: {type(data)}")
            log.info(f"Data device: {data.device}")

        if i >= (1000 - 1):
            break

    end_time = time.time()
    log.info(f"Time taken for 1000 iterations: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()



