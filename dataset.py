import tensorflow as tf
from pathlib import Path
import tempfile
import numpy as np

# from dcgan.configuration import Configuration


AUTOTUNE = tf.data.experimental.AUTOTUNE


class DatasetPipeline:
    def __init__(self, path, img_size, name, batch_size):
        self.path = path
        self.img_size = img_size
        self.dataset_name = name
        self.batch_size = batch_size
        # self.path = Configuration.DATASET_PATH
        # self.img_size = Configuration.IMAGE_SIZE
        # self.dataset_name = Configuration.DATASET_NAME
        self.num_images = None
        self.label_strings = None

    def load_dataset(self):
        ds, num_images, label_strings = self._load_data()

        self.num_images = num_images
        self.label_strings = label_strings

        ds = ds.map(lambda im, label: self.preprocess_image(im, label), AUTOTUNE)
        ds = self.dataset_cache(ds)
        ds = ds.shuffle(buffer_size=num_images * 10)
        # ds = ds.shuffle(buffer_size=num_images, reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
        # ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        # ds = ds.batch(Configuration.BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
        return ds

    def preprocess_image(self, image, label):
        image = tf.image.resize(image, (self.img_size[0], self.img_size[1]), antialias=True)
        image = (tf.dtypes.cast(image, tf.float32) / 127.5) - 1.0
        return image, label

    def _load_data(self):
        dataset = np.load(self.path)
        images = dataset['images']
        # print(images.shape)
        labels = dataset['labels']
        label_strings = dataset['str_labels']

        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        return ds, images.shape[0], label_strings

    def dataset_cache(self, dataset):
        tmp_dir = Path(tempfile.gettempdir())
        cache_dir = tmp_dir.joinpath('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        for p in cache_dir.glob(self.dataset_name + '*'):
            p.unlink()
        return dataset.cache(str(cache_dir / self.dataset_name))

    def get_num_labels(self):
        return self.label_strings.shape[0]

    def get_label_strings(self):
        return self.label_strings
