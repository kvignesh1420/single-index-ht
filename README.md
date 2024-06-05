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

### Single config runs

Probe the weights, features, overlap matrices etc for a single run.

Run using: 
```bash
(.venv) $ python main.py configs/main.yml
```

Run with learning rate schedule 
```bash
(.venv) $ python main.py configs/main_lr_schedule.yml
```

### Varying learning rates for GD/Adam

Experiments with multiple runs to plot the losses, KTA and PL Alphas for varying learning rates and optimizers.

Run using:
```bash
(.venv) $ python bulk_lr.py configs/bulk_lr.yml
```

### Losses with varying parameters

Experiments with multiple runs to plot the losses for varying parameters that one can set in the context.

Run with varying dataset size: `n`
```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_n.yml
```

Run with varying regularization parameter for regression: `reg_lambda`
```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_reg_lambda.yml
```

Run with varying label noise: `label_noise_std`
```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_label_noise_std.yml
```

Run with varying decay factor of StepLR learning rate schedule: `gamma`
```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_step_lr_gamma.yml
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.
