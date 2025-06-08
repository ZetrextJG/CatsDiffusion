# Cats Diffusion

Diffusion model trained on 30k images of cats.

This repository utilizes [Hydra](https://hydra.cc/docs/intro/) for config managment and is build on top of [guided_diffusion](https://github.com/openai/guided-diffusion/).
To run training run `python src/train_diffsuion.py` from the repository root. 
All evaluation methods are implemented in separate files in `./src` which start with `eval`, corresponding configs can be found in `./configs`.
