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

from solvers.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
from solvers.sa_solver import NoiseScheduleVP as SAScheduleVP
from solvers.sa_solver import SASolver

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
def compute_fid(sample_images, dataloader, num_samples=2048, micro_batch_size=32):
    """
    sample_images: function that takes a number of images and returns generated images
    """
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
            generated = sample_images(micro_batch_img.shape[0])
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

    eval_diffusion_orig = instantiate(config.diffusion, steps=1000)
    if config.exp.nfe != 1000:
        eval_diffusion = instantiate(config.diffusion, steps=1000, rescale_timesteps=True, timestep_respacing=str(config.exp.nfe))
    else:
        eval_diffusion = instantiate(config.diffusion, steps=1000)

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
    def sample_images(num_images):
        with ema.average_parameters():
            model.eval()
            match config.exp.sampling_method:
                case "ddim":
                    generated = eval_diffusion.ddim_sample_loop(model, (num_images, 3, 64, 64), progress=True)
                case "ddpm":
                    generated = eval_diffusion.ddim_sample_loop(model, (num_images, 3, 64, 64), eta=1.0, progress=True)
                case "dpm":
                    betas = torch.from_numpy(eval_diffusion_orig.betas).to(fabric.device)
                    schedule = NoiseScheduleVP("discrete", betas=betas)
                    model_fn = model_wrapper(model, schedule)
                    solver = DPM_Solver(model_fn, schedule)
                    noise = torch.randn((num_images, 3, 64, 64), device=fabric.device)
                    generated = solver.sample(noise, steps=config.exp.nfe) # Respacing happens in the solver
                case "sa":
                    betas = torch.from_numpy(eval_diffusion_orig.betas).to(fabric.device)
                    schedule = SAScheduleVP("discrete", betas=betas)
                    model_fn = model_wrapper(model, schedule)
                    solver = SASolver(model_fn, schedule)
                    noise = torch.randn((num_images, 3, 64, 64), device=fabric.device)
                    tau_func = lambda t: 1 # Fully stochastic sampling
                    generated = solver.sample("more_steps", noise, steps=config.exp.nfe, tau=tau_func)
                case _:
                    raise ValueError(f"Unknown sampling method: {config.exp.sampling_method}")
        model.train()
        return generated

    log.info("Starting evaluation...")

    with torch.inference_mode():
        log.info("Generating initial images for logging...")
        generated = sample_images(32)
        grid = make_grid(generated, nrow=8, normalize=True, value_range=(-1, 1))
        wandb.log({"generated": wandb.Image(grid)})

        log.info("Calculating FID score...")
        fid_score = compute_fid(sample_images, dataloader, num_samples=2048, micro_batch_size=config.exp.micro_batch_size)
        wandb.log({"fid": fid_score})


if __name__ == "__main__":
    main()

