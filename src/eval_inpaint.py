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

from corruption.inpaint import build_inpaint_freeform

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


@hydra.main(version_base = None, config_path = "../configs", config_name = "eval_inpaint")
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

    if config.exp.nfe != 1000:
        eval_diffusion = instantiate(config.diffusion, steps=1000, rescale_timesteps=True, timestep_respacing=str(config.exp.nfe))
    else:
        eval_diffusion = instantiate(config.diffusion, steps=1000)

    log.info("Diffusion and timesteps schedule initialized")

    inpaint_degrad = build_inpaint_freeform("freeform2030")

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

    log.info("Starting evaluation...")

    @torch.inference_mode()
    def restore_images(degraded, mask):
        with ema.average_parameters():
            if config.exp.algorithm == "ddrm":
                xt = degraded + torch.randn_like(degraded)
                for i in list(range(eval_diffusion.num_timesteps))[::-1]:
                    t = torch.tensor([i] * xt.shape[0], device=xt.device)
                    xt = eval_diffusion.ddrm_inpaint_sample(
                        model, xt, t, degraded, mask
                    )["sample"]
                return degraded * (1 - mask) + xt.clamp(-1, 1) * mask
            elif config.exp.algorithm == "ddnm":
                generated = eval_diffusion.ddim_sample_loop(
                    model, 
                    image_sample.shape, 
                    denoised_fn=lambda x0: degraded * (1 - mask) + x0 * mask,
                    eta=1.0,
                    progress=True, 
                )
                return generated
            else:
                raise ValueError(f"Unknown algorithm: {config.exp.algorithm}")


    with torch.inference_mode():
        log.info("Generating initial images for logging...")
        # generated = sample_images(32)
        # grid = make_grid(generated, nrow=8, normalize=True, value_range=(-1, 1))
        # wandb.log({"generated": wandb.Image(grid)})

        image_sample = next(iter(dataloader))[1][:8].to(fabric.device)
        degraded, mask = inpaint_degrad(image_sample)
        generated = restore_images(degraded, mask)
        all_images = torch.cat([image_sample, degraded, generated], dim=0)
        grid = make_grid(all_images, nrow=8, normalize=True, value_range=(-1, 1))
        wandb.log({"inpainting": wandb.Image(grid)})


        num_samples = 2048
        fid = FrechetInceptionDistance(feature=2048, normalize=True)

        original_images = []
        generated_images = []
        images_counter = 0
        pbar = tqdm(total=num_samples, desc="Computing FID", unit="images")
        for i, (idx, batch_img) in enumerate(dataloader):
            for micro_batch_img in torch.split(batch_img, config.exp.micro_batch_size):

                # Get the original images
                original_images.append(micro_batch_img.cpu())

                # Generate images
                degraded, mask = inpaint_degrad(micro_batch_img)
                generated = restore_images(degraded, mask)
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
        wandb.log({"fid": fid_score})


if __name__ == "__main__":
    main()

