import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.fabric import Fabric
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid
from torchinfo import summary
from pathlib import Path
from torch_ema import ExponentialMovingAverage

from utils.data_utils import InfiniteSampler
from utils.hydra_utils import set_log_dir
from utils.wandb_utils import setup_wandb

from torchmetrics.image.fid import FrechetInceptionDistance


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
    model: torch.nn.Module = instantiate(config.model)
    model = fabric.setup(model)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.exp.ema_decay)

    if config.exp.ckpt_path is not None:
        log.info(f"Loading model from {config.exp.ckpt_path}")
        checkpoint = fabric.load(config.exp.ckpt_path)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])

    return model,  ema


@torch.inference_mode()
def compute_fid(model, eval_diffusion, dataloader, num_samples=2048, micro_batch_size=32):
    fid = FrechetInceptionDistance(feature=2048, normalize=True)

    original_images = []
    generated_images = []

    images_counter = 0
    pbar = tqdm(total=num_samples, desc="Computing FID", unit="images")
    for i, (idx, batch_img) in enumerate(dataloader):
        for micro_batch_img in torch.split(batch_img, micro_batch_size):

            # Get the original images
            original_images.append(micro_batch_img.cpu())

            # Generate images
            generated = eval_diffusion.ddim_sample_loop(model, (micro_batch_img.shape[0], 3, 64, 64))
            generated_images.append(generated.cpu())

            images_counter += micro_batch_img.shape[0]
            pbar.update(micro_batch_img.shape[0])

        if images_counter >= num_samples:
            break

    original_images = torch.cat(original_images, dim=0)[:num_samples]
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]

    original_images = (original_images + 1) / 2 # Normalize to [0, 1]
    generated_images = (generated_images + 1) / 2 # Normalize to [0, 1]

    fid.update(original_images, real=True)
    fid.update(generated_images, real=False)

    fid_score = fid.compute().item()
    del fid
    torch.cuda.empty_cache()

    return fid_score


@hydra.main(version_base = None, config_path = "../configs", config_name = "eval_fid")
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

    eval_diffusion = instantiate(config.diffusion, steps=1000, rescale_timesteps=True)
    log.info("Diffusion and timesteps schedule initialized")

    model, ema = get_modules(config, fabric)
    log.info("Diffusion model initialized")

    log.info(f"""Model summary: """)
    # Run a model once on the same input shape as for training and generation to 
    # get the model compiled and get the summary
    input_shape = tuple(dataloader.dataset[0][1].shape)
    input_shape = (config.exp.micro_batch_size, *input_shape)
    summary(model, input_data=(
        torch.randn(input_shape).to(fabric.device),
        torch.Tensor([0] * input_shape[0]).to(fabric.device)
    ))

    @torch.inference_mode()
    def generate_and_log_images():
        with ema.average_parameters():
            model.eval()
            generated = eval_diffusion.ddim_sample_loop(model, (32, 3, 64, 64), progress=True)
            grid = make_grid(generated, nrow=8, normalize=True, value_range=(-1, 1))
            wandb.log({"generated": wandb.Image(grid)})
        model.train()

    @torch.inference_mode()
    def calculate_and_log_fid():
        with ema.average_parameters():
            model.eval()
            fid_score = compute_fid(model, eval_diffusion, dataloader, num_samples=2048, micro_batch_size=config.exp.micro_batch_size)
            wandb.log({"fid": fid_score})
        model.train()


    log.info("Starting evaluation...")

    generate_and_log_images()
    calculate_and_log_fid()


if __name__ == "__main__":
    main()

