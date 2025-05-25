import os
import tensorflow as tf
import scipy.io
from config import *
import numpy as np 

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Pascal3DDataset:
    def __init__(self, mode=MODE_TRAIN, image_size=IMAGE_SIZE, shuffle=False, augment=False):
        self.mode = mode
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment 

        self.image_dir = os.path.join(DATA_DIR, mode, 'images')
        self.ann_dir = os.path.join(DATA_DIR, mode, 'annotations')

        self.metadata = self._load_metadata()

    def _load_metadata(self):
        files = [f for f in os.listdir(self.ann_dir) if f.endswith('.mat')]
        image_paths = []
        bboxes = []
        azimuths = []

        for file in files:
            path = os.path.join(self.ann_dir, file)
            data = scipy.io.loadmat(path)

            try:
                record = data['record']
                filename = record['filename'][0][0][0]

                for obj in record['objects'][0][0][0]:
                    if obj['viewpoint'].size == 0:
                        continue

                    bbox = obj['bbox'][0].astype('float32')
                    azimuth = float(obj['viewpoint']['azimuth'][0][0][0][0])
                    image_path = os.path.join(self.image_dir, filename)

                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        bboxes.append(bbox)
                        azimuths.append(azimuth)
                        break
                    else:
                        print(f"Image file {image_path} does not exist.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        return (image_paths, bboxes, azimuths)
    
    def get_dataset(self, task):
        ds = tf.data.Dataset.from_tensor_slices(self.metadata)
        
        def preprocess(image_path, bbox, azimuth):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image.set_shape([None, None, 3])

            width = tf.cast(tf.shape(image)[1], tf.float32)
            height = tf.cast(tf.shape(image)[0], tf.float32)

            image = tf.image.resize(image, [self.image_size, self.image_size])
            image = tf.cast(image, tf.float32)
            
            bbox = tf.cast(bbox, tf.float32)
            bbox_normalized = tf.stack([
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height
            ])

            azimuth = tf.math.floormod(azimuth, 360.0)
            bin_index = tf.cast(azimuth // BIN_WIDTH, tf.int32)
            angle_rad = tf.multiply(azimuth, tf.constant(np.pi / 180.0))
            sin_val = tf.sin(angle_rad)
            cos_val = tf.cos(angle_rad)
            sin_cos = tf.cast(tf.stack([sin_val, cos_val]), tf.float32)
            
            if self.augment:
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

            if task == LOCALIZATION_TASK:
                return image, bbox_normalized
            elif task == ANGLE_CLASSIFICATION_TASK:
                return image, bin_index
            elif task == ANGLE_REGRESSION_TASK:
                return image, sin_cos
            elif task == MTL_REGRESSION:
                return image, {
                    LOCALIZATION_TASK: bbox_normalized,
                    ANGLE_REGRESSION_TASK: sin_cos
                }
            elif task == MTL_CLASSIFICATION:
                return image, {
                    LOCALIZATION_TASK: bbox_normalized,
                    ANGLE_CLASSIFICATION_TASK: bin_index
                }
            else: 
                return image, {
                    "bbox": bbox_normalized,
                    "azimuth": azimuth
                }
    
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.shuffle:
            ds = ds.shuffle(buffer_size=100, seed=RANDOM_SEED)
            
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

def adapt_for_uncertainty_mtl(dataset, task):
    def reformat_inputs(image, targets):
        if task == MTL_CLASSIFICATION:
            angle_class_one_hot = tf.one_hot(targets[ANGLE_CLASSIFICATION_TASK], BIN_COUNT)
            return ((image, targets[LOCALIZATION_TASK], angle_class_one_hot), tf.zeros((1,)))
        elif task == MTL_REGRESSION:
            return ((image, targets[LOCALIZATION_TASK], targets[ANGLE_REGRESSION_TASK]), tf.zeros((1,)))
        else:
            raise ValueError("Uncertainty loss only supported for MTL tasks.")
    
    return dataset.map(reformat_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    
if __name__ == "__main__":
    dataset = Pascal3DDataset()
    ds = dataset.get_dataset('classification')
    for images, targets in ds.take(1):
        print("Batch of images shape:", images.shape)
        print("Batch of targets:", targets)
        break
