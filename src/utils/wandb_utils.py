import omegaconf
import wandb


def setup_wandb(config):
    group, name = str(config.exp.log_dir).split('/')[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True
    )

    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        dir=config.exp.log_dir,
        group=group,
        name=name,
        config=wandb_config,
        sync_tensorboard=True
    )

    return run