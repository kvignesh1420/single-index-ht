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
        "n_test": 200,
        "batch_size": 2000,
        "h": 1500,
        "d": 1000,
        "label_noise_std": 0.3,
        "tau": 0.2,
        "num_epochs": 1,
        "optimizer": "sgd",
        "momentum": 0,
        "weight_decay": 0,
        "lr": 2000,
        "probe_freq": 1
    }
```

- To run the experiment, use:
```bash
(.venv) $ python main.py
```

- The outputs are generated in the `out/` folder based on a hash value corresponding to the context.