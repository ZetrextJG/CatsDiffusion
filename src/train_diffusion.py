import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.fabric import Fabric
import time
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

        if not config.diffusion.learn_sigma:
            # Remove learned sigma channels 
            checkpoint["out.2.weight"] = checkpoint["out.2.weight"][:3]
            checkpoint["out.2.bias"] = checkpoint["out.2.bias"][:3]

        model.load_state_dict(checkpoint)
    else:
        log.info("No checkpoint found, starting from scratch")

    return model, optimizer, ema


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
    eval_diffusion = instantiate(config.diffusion, steps=100, rescale_timesteps=True)
    schedule_sampler = instantiate(config.schedule_sampler)(diffusion=diffusion)
    log.info("Diffusion and timesteps schedule initialized")

    model, optimizer, ema = get_modules(config, fabric)
    log.info("Diffusion model initialized")

    log.info(f"""Model summary: """)
    # Run a model once on the same input shape as for training and generation to 
    # get the model compiled and get the summary
    input_shape = tuple(dataloader.dataset[0][1].shape)
    input_shape = (config.exp.micro_batch_size, *input_shape)
    print(summary(model, input_data=(
        torch.randn(input_shape).to(fabric.device),
        torch.Tensor([0] * input_shape[0]).to(fabric.device)
    )))


    def save_model(iter: int):
        model.eval()
        save_dir = Path(config.exp.log_dir) / "checkpoints"
        iter_k = iter // 1000
        fabric.save(
            save_dir / f"model_{iter_k}k.ckpt",
            {"model": model, "optimizer": optimizer, "ema": ema},
        )
        model.train()

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


    log.info("Starting to iterate over dataloader")
    pbar = tqdm(desc="Training", unit="steps")
    batch_mult = config.dataloader.batch_size // config.exp.micro_batch_size

    model.train()
    for i, (idx, batch_img) in enumerate(dataloader):
        optimizer.zero_grad()

        timestep_history = []
        loss_history = []

        for micro_batch_img in torch.split(batch_img, config.exp.micro_batch_size):
            t, weights = schedule_sampler.sample(micro_batch_img.shape[0], device=fabric.device)
            losses = diffusion.training_losses(model, micro_batch_img, t, model_kwargs=None)

            timestep_history.append(t.cpu())
            loss_history.append(losses["loss"].detach().cpu())

            loss = (losses["loss"] * weights).mean()
            fabric.backward(loss / batch_mult, model=model)

        optimizer.step()
        optimizer.zero_grad()
        ema.update(model.parameters())

        times_h = torch.concat(timestep_history, dim=0)
        losses_h = torch.concat(loss_history, dim=0)
        schedule_sampler.update_with_all_losses(times_h, losses_h)

        wandb.log({"loss": loss.item()}, step=i)
        pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)

        if i > config.exp.max_iters:
            log.info(f"Reached max iterations: {config.exp.max_iters}")
            generate_and_log_images()
            calculate_and_log_fid()
            save_model(i)
            break

        if i % 1000 == 0:
            generate_and_log_images()

        if i % 5000 == 0:
            calculate_and_log_fid()
            if i != 0:
                save_model(i)

    pbar.close()


if __name__ == "__main__":
    main()

