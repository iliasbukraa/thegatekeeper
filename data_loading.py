from pathlib import Path
import pandas as pd

masked_images = Path('data/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset')
nonmask_images = Path('data/self-built-masked-face-recognition-dataset/AFDB_face_dataset')
mask_dataframe = pd.DataFrame()

for folder in list(masked_images.iterdir()):
    for image_path in folder.iterdir():
        mask_dataframe = mask_dataframe.append({
            'image': str(image_path),
            'mask': 1
        }, ignore_index=True)

for folder in list(nonmask_images.iterdir()):
    for image_path in folder.iterdir():
        mask_dataframe = mask_dataframe.append({
            'image': str(image_path),
            'mask': 0
        }, ignore_index=True)

mask_dataframe.to_pickle('mask_df.pickle')
