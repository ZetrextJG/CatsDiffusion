import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.fabric import Fabric
import time
import wandb
from tqdm import tqdm

from torch_ema import ExponentialMovingAverage

from utils.data_utils import InfiniteSampler
from utils.hydra_utils import set_log_dir
from utils.wandb_utils import setup_wandb
from torchinfo import summary


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


def get_modules(config, fabric: Fabric):
    model = instantiate(config.model)
    optimizer = instantiate(config.optimizer)(params=model.parameters())
    model, optimizer = fabric.setup(model, optimizer)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.exp.ema_decay)

    if config.exp.ckpt_path is not None:
        log.info(f"Loading model from {config.exp.ckpt_path}")
        checkpoint = fabric.load(config.exp.ckpt_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
    elif config.exp.pretrained_ckpt_path is not None:
        log.info(f"Loading pretrained model from {config.exp.pretrained_ckpt_path}")
        checkpoint = fabric.load(config.exp.pretrained_ckpt_path)
        checkpoint.pop("label_emb.weight") # Remove class conditioning
        model.load_state_dict(checkpoint)
    else:
        log.info("No checkpoint found, starting from scratch")
    return model, optimizer, ema


@hydra.main(version_base = None, config_path = "../configs", config_name = "train_diffusion")
def main(config: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision("high")

    config = set_log_dir(config)
    log.info(f"Log dir: {config.exp.log_dir}")

    run = setup_wandb(config)
    log.info("WandB initialized")

    fabric: Fabric = get_fabric(config)
    log.info("Fabric initialized")

    dataloader = get_dataloader(config, fabric)
    log.info("Dataloader initialized")

    diffusion = instantiate(config.diffusion)
    schedule_sampler = instantiate(config.schedule_sampler)(diffusion=diffusion)
    log.info("Diffusion and timesteps schedule initialized")

    model, optimizer, ema = get_modules(config, fabric)
    log.info("Diffusion model initialized")

    input_shape = tuple(dataloader.dataset[0][1].unsqueeze(0).shape) # (1, 3, 64, 64)
    log.info(f"""Model summary: """)
    print(summary(model, input_data=(
        torch.randn(input_shape).to(fabric.device), torch.Tensor([0]).to(fabric.device)
    )))

    log.info("Starting to iterate over dataloader")
    pbar = tqdm(desc="Training", unit="steps")
    for i, (idx, batch_img) in enumerate(dataloader):

        optimizer.zero_grad()
        t, weights = schedule_sampler.sample(batch_img.shape[0], device=fabric.device)
        losses = diffusion.training_losses(model, batch_img, t, model_kwargs=None)
        loss = (losses["loss"] * weights).mean()
        fabric.backward(loss)

        optimizer.step()
        ema.update(model.parameters())

        wandb.log({"loss": loss.item()})
        pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)

    pbar.close()



if __name__ == "__main__":
    main()

