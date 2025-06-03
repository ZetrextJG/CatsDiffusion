import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.fabric import Fabric
import wandb
from torchvision.utils import make_grid
from torchinfo import summary
from torch_ema import ExponentialMovingAverage
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


@hydra.main(version_base = None, config_path = "../configs", config_name = "eval_trajectory")
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

            log_steps = np.linspace(0, eval_diffusion.num_timesteps - 1, config.exp.log_steps).astype(int)
            log_steps = set(log_steps.tolist())

            x_preds = []
            x0_preds = []
            for i, out in enumerate(eval_diffusion.ddim_sample_loop_progressive(model, (config.exp.micro_batch_size, 3, 64, 64), progress=True)):
                if i in log_steps:
                    x_preds.append(out["sample"].cpu())
                    x0_preds.append(out["pred_xstart"].cpu())

            x_preds = torch.stack(x_preds, dim=0)
            x0_preds = torch.stack(x0_preds, dim=0)

            stacked = torch.stack([x_preds, x0_preds], dim=2)  # (steps, batch_size, 2,  3, 64, 64)
            stacked = stacked.transpose(0, 1) # Make time on x axis, batch on y axis
            stacked = stacked.transpose(1, 2).reshape(-1, 3, 64, 64)
            grid = make_grid(stacked, nrow=config.exp.micro_batch_size, normalize=True, value_range=(-1, 1))
            wandb.log({"trajectory": wandb.Image(grid)})


if __name__ == "__main__":
    main()

