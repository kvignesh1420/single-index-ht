# Crafting Heavy-Tails in Weight Matrix Spectrum without Gradient Noise

Explore various ways of generating heavy tails in the weight matrix spectrum without gradient noise. In particular, we train shallow neural networks
with full batch GD/Adam and large learning rates for multiple steps.

## Setup

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

1. `main.py`: Script used to probe the weights, features, overlap matrices etc for a single run.

Example:
```bash
(.venv) $ python main.py configs/main.yml

# execute with learning rate schedule
(.venv) $ python main.py configs/main_lr_schedule.yml
```

2. `bulk_lr.py`: Script used for faster experiments with multiple runs to plot the losses, KTA and PL Alphas for varying learning rates and optimizers.

Example:
```bash
(.venv) $ python bulk_lr.py configs/bulk_lr.yml
```

3. `bulk_losses.py`: Script used for faster experiments with multiple runs to plot the losses for varying parameters that one can set in the context.

Example:
```bash
# vary dataset size: `n`
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_n.yml

# vary regularization parameter for regression: `reg_lambda`
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_reg_lambda.yml

# vary label noise: `label_noise_std`
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_label_noise_std.yml

# vary decay factor of StepLR learning rate schedule: `gamma`
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_step_lr_gamma.yml
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.
