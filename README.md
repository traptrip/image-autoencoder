# nn-image-compression

# Inference

## Encode image
```bash
python main.py encode -m <model_path> -src <path/to/image> -dsc <path/to/latent/vector>
```

## Decode image
```bash
python main.py decode -m <model_path> -src <path/to/latent/vector> -dsc <path/to/image>
```

# Training
## Prepare dataset
To train from scratch you need to download imagenet dataset from kaggle
1. Go to [competition page](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview) and accept its rules
2. run `cd data && sh download_imagenet.sh`

If you have problems using kaggle api check https://www.kaggle.com/docs/api. 

## Configure training parameters
[./config/config.yaml](./config/config.yaml) all the parameters for training. Parameters are configurating using Hydra.

## Train AutoEncoder
```python
python train.py
```
Training logs will be saved in `runs/basic_autoencoder` directory by default

To see the training plots use tensorboard

```
tensorboard --logdir <path/to/experiments/directory>
```
