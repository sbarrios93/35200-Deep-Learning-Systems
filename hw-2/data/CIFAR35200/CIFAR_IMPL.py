"""CIFAR35200 dataset."""
import numpy as np
import tensorflow_datasets as tfds

from typing import Union
# from pathlib import Path

class Cifar35200(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for CIFAR35200 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Download the original CIFAR10 dataset from `https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz`
  Download the CIFAR10.1 v6 test dataset from 
    - `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy`
    - `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy`
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(CIFAR35200): Specifies the tfds.core.DatasetInfo object
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(32, 32, 3)),
                'label': tfds.features.ClassLabel(num_classes=10),
            }),
        supervised_keys=("image", "label"),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(CIFAR35200): read the data and defines the splits
    # In most cases, datasets are downloaded upfront, and some preprocessing might already have been performed on them.
    # so here we mimic this scenario by specifying the existing archive locations
    # Be sure to remember to build this CIFAR35200 dataset by `tfds build --manual_dir pathToExistingArchives`
    # Read MANUAL_DOWNLOAD_INSTRUCTIONS above. This variable is required if using manual download and extraction
    # To perform direct download and extraction, please refer to:
    #   - https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/cifar10_1.py
    #   - https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/cifar.py
    """Returns SplitGenerators."""
    
    # training & validation sets: original cifar10
    cifar_path = dl_manager.manual_dir / "cifar-10-batches-bin"
    
    # test set: cifar10.1 v6
    cifar_101_image = dl_manager.manual_dir / "cifar10.1_v6_data.npy"
    cifar_101_label = dl_manager.manual_dir / "cifar10.1_v6_labels.npy"
    
    # extract and update the real label names
    label_path = cifar_path / 'batches.meta.txt'
    label_names = [name for name in label_path.read_text().split('\n') if name]
    self.info.features['label'].names = label_names
    
    train_files = [f'data_batch_{i}.bin' for i in range(1,6)]
    train_paths = [cifar_path / f for f in train_files]
    valid_files = ['test_batch.bin']
    valid_paths = [cifar_path / f for f in valid_files]
    
    return {
        "train": self._generate_examples("train", train_paths),
        "valid": self._generate_examples("valid", valid_paths),
        "test": self._generate_examples("test", [cifar_101_image, cifar_101_label]),
    }

  def _generate_examples(self, mode, filepaths) -> Union[int, dict]:
    """Yields examples."""
    # TODO(CIFAR35200): Yields (key, example) tuples from the dataset
    # Type hint is recommended here if you have special handling for different split.
    # Why? it took me some time to figure out adding `.squeeze()` in _load_data()
    # From Python>=3.10, you can do `fn(...) -> int | dict:`, but colab has 3.7
    """Yields examples."""
    if mode == "test":
        images = np.load(filepaths[0])
        labels = np.load(filepaths[1])
        for i, (label, image) in enumerate(zip(labels, images)):
            record = {
                "image": image,
                "label": label,
            }
            yield i, record
    else:
        index = 0
        for path in filepaths:
            for label, image in _load_data(path):
                record = {
                    "image": image,
                    "label": label,
                }
                yield index, record
                index += 1

# helper functions for parsing CIFAR10 original binary data
def _load_data(path):
    data = path.read_bytes()
    offset = 0
    max_offset = len(data) - 1
    while offset < max_offset:
        # label: 1st byte of every 3073 bytes from binary buffer -> numpy array (1,) -> int
        label = np.frombuffer(data, dtype=np.uint8, count=1, offset=offset).squeeze()
        offset += 1
        img = (np.frombuffer(data, dtype=np.uint8, count=3072, offset=offset).reshape((3, 32, 32)).transpose((1 ,2, 0)))
        offset += 3072
        yield label, img