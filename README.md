# Exploring heavy tails in single index models

This effort aims to analyze the role of heavy tails in the ESD of weight matrices of two and three layer fully-connected neural networks.

## Setup

```bash
$ python3.9 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Experiments

- As of now, modify the `exp_context` dictionary in `main.py` to configure the experiment. A sample value is shown below:
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
        "tau": 0.2,
        "num_epochs": 1,
        "optimizer": "adam",
        "momentum": 0,
        "weight_decay": 0,
        "lr": 1,
        "reg_lamba": 0.01,
        "enable_weight_normalization": True,
        # NOTE: The probing now occurs based on number of steps.
        # set appropriate values based on n, batch_size and num_epochs.
        "probe_freq_steps": 1
    }
```

- To run the experiment, use:
```bash
(.venv) $ python main.py
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.