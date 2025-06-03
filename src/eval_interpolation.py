import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.fabric import Fabric
import wandb
from torchvision.utils import make_grid
from torchinfo import summary
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import numpy as np

from utils.hydra_utils import set_log_dir
from utils.wandb_utils import setup_wandb



import logging
log = logging.getLogger(__name__)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


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


@hydra.main(version_base = None, config_path = "../configs", config_name = "eval_interpolation")
def main(config: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision("high")

    config = set_log_dir(config)
    log.info(f"Log dir: {config.exp.log_dir}")

    run = setup_wandb(config)
    log.info("WandB initialized")

    fabric: Fabric = get_fabric(config)
    log.info("Fabric initialized")

    if config.exp.nfe != 1000:
        eval_diffusion = instantiate(config.diffusion, steps=1000, rescale_timesteps=True, timestep_respacing=str(config.exp.nfe))
    else:
        eval_diffusion = instantiate(config.diffusion, steps=1000)
    log.info("Diffusion and timesteps schedule initialized")

    model, ema = get_modules(config, fabric)
    log.info("Diffusion model initialized")

    with torch.inference_mode():

        with ema.average_parameters():
            model.eval()

            log.info(f"""Model summary: """)
            # Run a model once on the same input shape as for training and generation to 
            # get the model compiled and get the summary
            input_shape = (3, 64, 64)
            input_shape = (config.exp.micro_batch_size, *input_shape)
            summary(model, input_data=(
                torch.randn(input_shape).to(fabric.device),
                torch.Tensor([0] * input_shape[0]).to(fabric.device)
            ))

            shape = (config.exp.micro_batch_size, 3, 64, 64)
            noise = torch.randn(shape, device=fabric.device)

            # Sample initial images
            eta = 0.0 if config.exp.ddim else 1.0
            generated = eval_diffusion.ddim_sample_loop(model, shape, noise=noise, progress=True, eta=eta)
            generated_grid = make_grid(generated, nrow=config.exp.micro_batch_size, normalize=True, value_range=(-1, 1))
            wandb.log({"generated": wandb.Image(generated_grid)})

            noise_grid = make_grid(noise, nrow=config.exp.micro_batch_size, normalize=True, value_range=(-1, 1))
            wandb.log({"noise": wandb.Image(noise_grid)})

            # Interpolate between the two generated images
            noise2 = torch.roll(noise, shifts=-1, dims=0).unsqueeze(0) # (1, BS)
            noise = noise.unsqueeze(0) # (1, BS)

            noise = noise.repeat_interleave(config.exp.interpolation_steps, dim=0) # (IS, BS)
            noise2 = noise2.repeat_interleave(config.exp.interpolation_steps, dim=0) # (IS, BS)

            lambdas = torch.linspace(0, 1, config.exp.interpolation_steps, device=fabric.device)
            lambdas = lambdas.view(-1, 1, 1, 1, 1)

            interpolated_noise = noise * (1 - lambdas) + noise2 * lambdas  # (IS, BS, 3, 64, 64)
            interpolated_noise = interpolated_noise.reshape(-1, 3, 64, 64)  # (IS * BS, 3, 64, 64)

            interpolated = eval_diffusion.ddim_sample_loop(model, tuple(interpolated_noise.shape), noise=interpolated_noise, progress=True, eta=eta)
            interpolated = interpolated.reshape(config.exp.interpolation_steps, config.exp.micro_batch_size, 3, 64, 64)  # (IS, BS, 3, 64, 64)
            interpolated = interpolated.transpose(0, 1).reshape(-1, 3, 64, 64)  # (BS * IS, 3, 64, 64)
            interpolated_grid = make_grid(interpolated, nrow=config.exp.interpolation_steps, normalize=True, value_range=(-1, 1))
            wandb.log({"interpolated": wandb.Image(interpolated_grid)})


if __name__ == "__main__":
    main()

