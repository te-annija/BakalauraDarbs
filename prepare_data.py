import os
import shutil
import random
import scipy.io
import numpy as np
from collections import defaultdict
from config import *

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def get_object_azimuth(obj):
    if obj["viewpoint"].size == 0:
        return None
    return float(obj["viewpoint"]["azimuth"][0][0][0][0])

def is_one_car_per_image(file_path):
    data = scipy.io.loadmat(file_path)
    objects = data["record"]["objects"][0][0][0]
    valid_objects = [obj for obj in objects if obj["viewpoint"].size > 0]
    return len(valid_objects) == 1

def find_image_file(image_dir_path, file_basename):
    for ext in [".jpg", ".JPEG"]:
        img_path = os.path.join(image_dir_path, file_basename + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def get_file_with_azimuth_bin(annotation_dir_path, file_name):
    file_path = os.path.join(annotation_dir_path, file_name)
    data = scipy.io.loadmat(file_path)
    objects = data["record"]["objects"][0][0][0]

    for obj in objects:
        azimuth = get_object_azimuth(obj)
        if azimuth is not None:
            azimuth = np.mod(azimuth, 360.0)
            bin_index = int(azimuth // BIN_WIDTH)
            return bin_index

    return None

def split_data_with_balanced_bins(folder_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    annotation_dir_path = os.path.join(ANN_DIR_RAW, folder_name)
    image_dir_path = os.path.join(IMG_DIR_RAW, folder_name)

    files = [f for f in os.listdir(annotation_dir_path) if f.endswith(".mat")]
    files = [
        f for f in files if is_one_car_per_image(os.path.join(annotation_dir_path, f))
    ]

    files_by_bin = defaultdict(list)
    for file in files:
        bin_index = get_file_with_azimuth_bin(annotation_dir_path, file)
        if bin_index is not None:
            files_by_bin[bin_index].append(file)

    train_files = []
    val_files = []
    test_files = []

    for bin_index, bin_files in files_by_bin.items():
        random.shuffle(bin_files)
        bin_count = len(bin_files)

        bin_train_count = int(bin_count * train_ratio)
        bin_val_count = int(bin_count * val_ratio)

        train_files.extend(bin_files[:bin_train_count])
        val_files.extend(bin_files[bin_train_count : bin_train_count + bin_val_count])
        test_files.extend(bin_files[bin_train_count + bin_val_count :])

    for split, files in zip(MODES, [train_files, val_files, test_files]):
        bin_distribution = defaultdict(int)

        for file in files:
            file_basename = file.replace(".mat", "")
            img_file = find_image_file(image_dir_path, file_basename)

            if img_file is None:
                print(f"Warning: No image found for {file_basename} in {folder_name}")
                continue
            ann_file = os.path.join(annotation_dir_path, file)

            shutil.copy(
                img_file,
                os.path.join(DATA_DIR, split, "images", os.path.basename(img_file)),
            )
            shutil.copy(
                ann_file,
                os.path.join(DATA_DIR, split, "annotations", os.path.basename(file)),
            )

            bin_index = get_file_with_azimuth_bin(annotation_dir_path, file)
            if bin_index is not None:
                bin_distribution[bin_index] += 1

        print(f"\n{split.capitalize()} set bin distribution for {folder_name}:")
        num_bins = int(360 // BIN_WIDTH)
        for bin_idx in range(num_bins):
            count = bin_distribution.get(bin_idx, 0)
            print(
                f"  Bin {bin_idx} ({bin_idx * BIN_WIDTH}° to {(bin_idx + 1) * BIN_WIDTH}°): {count} images"
            )

    print(
        f"\nData split for {folder_name} completed: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files."
    )

    total_bins = defaultdict(int)
    for bin_idx, files in files_by_bin.items():
        total_bins[bin_idx] = len(files)

    print(f"\nOverall bin distribution for {folder_name}:")
    num_bins = int(360 // BIN_WIDTH)
    for bin_idx in range(num_bins):
        count = total_bins.get(bin_idx, 0)
        print(
            f"  Bin {bin_idx} ({bin_idx * BIN_WIDTH}° to {(bin_idx + 1) * BIN_WIDTH}°): {count} images"
        )


def main(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    for split in MODES:
        os.makedirs(os.path.join(DATA_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, "annotations"), exist_ok=True)

    for folder_name in RAW_DATA_FOLDERS:
        split_data_with_balanced_bins(folder_name, train_ratio, val_ratio, test_ratio)


if __name__ == "__main__":
    main()
