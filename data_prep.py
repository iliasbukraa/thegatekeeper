from pathlib import Path
import pandas as pd


dataset = Path('data/self-built-masked-face-recognition-dataset')
masked_pictures = dataset/'AFDB_masked_face_dataset'
nonmasked_pictures = dataset/'AFDB_face_dataset'
mask_dataframe = pd.DataFrame()

for directory in list(masked_pictures.iterdir()):
    for image_path in directory.iterdir():
        mask_dataframe = mask_dataframe.append({
            'image': str(image_path),
            'mask': 1
        }, ignore_index=True)

for directory in list(nonmasked_pictures.iterdir()):
    for image_path in directory.iterdir():
        nonmask_dataframe = mask_dataframe.append({
            'image': str(image_path),
            'mask': 1
        }, ignore_index=True)

dataframe_path = 'data/dataframe.pickle'
print(f'saving dataframe to {dataframe_path}')
mask_dataframe.to_pickle(dataframe_path)

