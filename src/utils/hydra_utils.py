from omegaconf import DictConfig, open_dict
from pathlib import Path

def set_log_dir(config: DictConfig) -> Path:
    '''
    Extracts path to output directory created by Hydra as pathlib.Path instance
    '''
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))
    output_dir = Path.cwd() / 'outputs' / date
    with open_dict(config):
        config.exp.log_dir = output_dir
    return config
