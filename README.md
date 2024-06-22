## Crafting Heavy-Tails in Weight Matrix Spectrum without Gradient Noise

This repository explores various methods to generate heavy tails in the weight matrix spectrum of neural networks without the influence of gradient noise. We specifically train shallow neural networks using full-batch Gradient Descent (GD) or Adam optimizer with large learning rates over multiple steps.

## Setup

To get started, set up your virtual environment and install the required dependencies:

```bash
$ python3.9 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

### Single Configuration Runs

Investigate the properties of weights, features, overlap matrices, and more for a single configuration:

```bash
(.venv) $ python main.py configs/main.yml
```

To run with a learning rate schedule:

```bash
(.venv) $ python main.py configs/main_lr_schedule.yml
```

### Varying Learning Rates for GD/Adam

Conduct experiments with multiple runs to plot losses, Kernel Target Alignment (KTA), and Power Law (PL) Alphas for different learning rates and optimizers:

```bash
(.venv) $ python bulk_lr.py configs/bulk_lr.yml
```

### Losses with Varying Parameters

Perform experiments with multiple runs to plot the losses for different parameter settings:

#### Varying Dataset Size: `n`

```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_n.yml
```

#### Varying Regularization Parameter for Regression: `reg_lambda`

```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_reg_lambda.yml
```

#### Varying Label Noise: `label_noise_std`

```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_label_noise_std.yml
```

#### Varying Decay Factor of `StepLR` Learning Rate Schedule: `gamma`

```bash
(.venv) $ python bulk_losses.py configs/bulk_losses_vary_step_lr_gamma.yml
```

### Output

The outputs of the experiments are stored in the `out/` directory, named according to a hash value based on the experiment context.

## Citation

```bibtex
@misc{kothapalli2024crafting,
      title={Crafting Heavy-Tails in Weight Matrix Spectrum without Gradient Noise}, 
      author={Vignesh Kothapalli and Tianyu Pang and Shenyang Deng and Zongmin Liu and Yaoqing Yang},
      year={2024},
      eprint={2406.04657},
      archivePrefix={arXiv},
}
```