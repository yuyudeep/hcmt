# HCMT: Learning Flexible Body Collision Dynamics with Hierarchical Mesh Transformer
The paper is available on [link](https://arxiv.org/abs/2312.12467).

## Set environment
The environment can be set up using either `environment.yaml` file or manually installing the dependencies.
### Using an environment.yaml file
```
conda env create -f environment.yaml
```

## Requirements

- tensorflow-gpu==2.8.0
- dm-sonnet==2.0.1
- protobuf==3.20.0

## Download datasets
We host the datasets on this [link](https://figshare.com/s/a2c4abb9872b1dae3286)
All data gets downloaded and stored in `data\impact` directory.


## How to run
To run each experiment, navigate into `HCMT-main`. Then, run the following command:

### Train a model:
```
python -m run_model --mode=train
```

### Generate trajectory rollouts:
```
python -m run_model --mode=eval
```
