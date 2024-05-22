# Exploring heavy tails in single index models

This effort aims to analyze the role of heavy tails in the ESD of weight matrices of two and three layer fully-connected neural networks.

## Setup

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

There are four files for running different kinds of experiments:
NOTE: GPU support will be added soon.

1. `main.py`: Script used to probe the weights, features, overlap matrices etc for a single run

- One can modify the `exp_context` dictionary in `main.py` to configure the experiment. A sample value is shown below:
```py
exp_context = {
    "L": 2,
    "n": 2000,
    "batch_size": 2000,
    "n_test": 200,
    "batch_size_test": 200,
    "h": 1500,
    "d": 1000,
    "label_noise_std": 0.3,
    "num_epochs": 10,
    "optimizer": "adam",
    "momentum": 0,
    "weight_decay": 0,
    "lr": 1,
    "reg_lambda": 1e-2,
    "enable_weight_normalization": False,
    # The probing occurs based on number of steps.
    # set appropriate values based on n, batch_size and num_epochs.
    "probe_freq_steps": 10,
    # probe weights to plot the ESD and vals.
    "probe_weights": True,
    # set "plot_overlaps": True for plotting the overlaps between singular vectors.
    # Note: make sure that "probe_freq_steps": 1 when this is set.
    "plot_overlaps": False,
    # probe features to plot the KTA
    "probe_features": True,
    "fix_last_layer": True,
    "enable_ww": False # setting `enable_ww` to True will open plots that need to be closed manually.
}
```

2. `main_lr_schedule.py`: Script used to probe the weights, features, overlap matrices etc for a single run using a learning rate schedule.

3. `bulk_lr.py`: Script used for faster experiments with multiple runs to plot the losses, KTA and PL Alphas for varying learning rates and optimizers

4. `bulk_losses.py`: Script used for faster experiments with multiple runs to plot the losses for varying parameters that one can set in the context.


- To run the experiment, simply run the script. For example:
```bash
(.venv) $ python main.py
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.
