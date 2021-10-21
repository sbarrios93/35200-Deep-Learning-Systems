import yaml
import tensorflow as tf
from src.model import run_transformer

print("Check GPU and TF Version")
print(f"{tf.config.list_physical_devices()=}")
print(f"{tf.__version__=}")

with open("conf.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

models = config["MODELS"]
for k, v in models.items():
    run_transformer(MODEL_NAME=k, **v)