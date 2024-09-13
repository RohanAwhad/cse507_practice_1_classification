import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
from tqdm import tqdm

# ===
# Constants
# ===
SAVE_DIR = "/scratch/rawhad/CSE507/practice_1/chexpert_preprocessed_ds/train_ds"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===
# Define transformation
# ===
class ChestXRayTransform:
    def __init__(self, size=320, rotation_range=(-5, 5), scale_range=(0.9, 1.1), 
                 translation_range=(-0.1, 0.1), zoom_range=(0.9, 1.1), 
                 brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1)):
        self.size = size
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img, only_resize: bool):

        if not only_resize:
            # Rotation
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            img = transforms.functional.rotate(img, angle)

            # Scaling
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            img = transforms.functional.resize(img, (int(img.size[0]*scale), int(img.size[1]*scale)))

            # Translation
            translation_x = random.uniform(self.translation_range[0], self.translation_range[1])
            translation_y = random.uniform(self.translation_range[0], self.translation_range[1])
            img = transforms.functional.affine(img, angle=0, translate=(int(img.size[0]*translation_x), int(img.size[1]*translation_y)), scale=1, shear=0)

            # Zoom
            zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])
            img = transforms.functional.resize(img, (int(img.size[0]*zoom), int(img.size[1]*zoom)))

            # Lighting adjustments
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
            contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
            img = transforms.functional.adjust_brightness(img, brightness)
            img = transforms.functional.adjust_contrast(img, contrast)

            # Resize 
            img = transforms.Resize(self.size)(img)

            # random location square crop of self.size
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.size, self.size))
            img = transforms.functional.crop(img, i, j, h, w)
        else:
            # Resize 
            img = transforms.Resize(self.size)(img)

            # center square crop of self.size
            w, h = img.size
            mid_w, mid_h = w//2, h//2
            i, j = mid_w - self.size//2, mid_h - self.size//2
            img = transforms.functional.crop(img, i, j, self.size, self.size)


        # to tensor
        img = transforms.ToTensor()(img)
        return img


transform = ChestXRayTransform()

# ===
# Process images
# ===
N_AUGMENTATIONS = 4
def process_image(path):
    img = Image.open(path)
    original_img = transform(img, only_resize=True)
    transformed_imgs = [transform(img, only_resize=False) for _ in range(N_AUGMENTATIONS)]
    return [original_img, ] + transformed_imgs

LABEL_COLS = 'Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices,No Finding'.split(',')
def process_images(paths):
    results = []
    for path in tqdm(paths):
        images = process_image(path)
        label_identifier = 'CheXpert-v1.0/train/' + '/'.join(path.split('/')[-3:])
        label = labels_df[labels_df['Path'] == label_identifier][LABEL_COLS].iloc[0].values.astype(np.float32)
        results.extend([(img, label) for img in images])
    return results

# ===
# Save results
# ===
def save_results(results, shard_id, save_dir):
    with open(os.path.join(save_dir, f"shard_{shard_id}.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print('Saved Shard:', shard_id, 'at', save_dir)

# ===
# Find all JPG files
# ===
import os
ORIGINAL_DS_ROOT_DIR = '/data/courses/2024/class_ImageSummerFall2024_jliang12/chexpertchestxrays-u20210408'
def find_jpg_files(root_dir):
    jpg_files = []
    for root, dirs, files in os.walk(os.path.join(ORIGINAL_DS_ROOT_DIR, root_dir)):
        for file_ in files:
            if file_.endswith('.jpg'): jpg_files.append(os.path.join(root, file_))
        if len(jpg_files) > 1000: print('Retrieved', len(jpg_files), 'paths uptill now')

    print('Root dir:', root_dir, 'done!')
    return jpg_files

if os.path.exists('all_paths.txt'):
    with open('all_paths.txt', 'r') as f: paths = [x.strip() for x in f.readlines() if x.strip()]
else:
    root_dirs = ('CheXpert-v1.0 batch 2 (train 1)', 'CheXpert-v1.0 batch 3 (train 2)', 'CheXpert-v1.0 batch 4 (train 3)')
    paths = []
    for x in root_dirs:
        paths.extend(find_jpg_files(x))
    with open('all_paths.txt', 'w') as f: f.write('\n'.join(paths))


print('Gotten all paths')
# ===
# Load labels
# ===

labels_df = pd.read_csv("/data/courses/2024/class_ImageSummerFall2024_jliang12/chexpertchestxrays-u20210408/train_cheXbert.csv").fillna(-1)
print(labels_df.head())


# ===
# Main
# ===
#paths = labels_df['Path'].values
#n_procs = max(1, os.cpu_count()//2)
n_procs = 8
SHARD_SIZE = 5000
print(f"Using {n_procs} processes")


with mp.Pool(n_procs) as pool:
    chunksize = 100
    results = []
    shard_id = 0
    for chunk in tqdm(pool.imap_unordered(process_images, [paths[i:i+chunksize] for i in range(0, len(paths), chunksize)], chunksize=1)):
        results.extend(chunk)
        if len(results) >= SHARD_SIZE:
            save_results(results, shard_id, SAVE_DIR)
            results = []
            shard_id += 1
    if len(results) > 0:
        save_results(results, shard_id, SAVE_DIR)



# ===
# Valid Split
# ===
print('Sharding Validation split')

VALID_SAVE_DIR = "/scratch/rawhad/CSE507/practice_1/chexpert_preprocessed_ds/valid_ds"
os.makedirs(VALID_SAVE_DIR, exist_ok=True)

valid_labels_df = pd.read_csv("/data/courses/2024/class_ImageSummerFall2024_jliang12/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_1_val_and_csv/valid.csv").fillna(-1)
print(valid_labels_df.head())

def process_valid_image(path):
    img = Image.open(path)
    img = transform(img, only_resize=True)
    label_identifier = 'CheXpert-v1.0/valid/' + '/'.join(path.split('/')[-3:])
    label = valid_labels_df[valid_labels_df['Path'] == label_identifier][LABEL_COLS].iloc[0].values.astype(np.float32)
    return (img, label)

valid_paths = []
for root, dirs, files in os.walk("/data/courses/2024/class_ImageSummerFall2024_jliang12/chexpertchestxrays-u20210408/CheXpert-v1.0_batch_1_val_and_csv/valid"):
    for file_ in files:
        if file_.endswith('.jpg'): valid_paths.append(os.path.join(root, file_))

valid_results = []
shard_id = 0
for path in tqdm(valid_paths):
    valid_results.append(process_valid_image(path))
    if len(valid_results) >= 5000:
        save_results(valid_results, shard_id, VALID_SAVE_DIR)
        valid_results = []
        shard_id += 1
if len(valid_results) > 0:
    save_results(valid_results, shard_id, VALID_SAVE_DIR)
