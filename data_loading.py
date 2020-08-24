'''
thsi script is run to create the dataframe containing both the image paths in one column and status (mask/nomask) in the
second column
'''


from pathlib import Path
import pandas as pd

# define the paths with the images used to train the face mask detection model
masked_images = Path('data/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset')
nonmask_images = Path('data/self-built-masked-face-recognition-dataset/AFDB_face_dataset')

mask_dataframe = pd.DataFrame()

# iterate over said folders and save the paths to each image and mask/no mask state in a pandas dataframe
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

# save the dataframe in a pickle format
mask_dataframe.to_pickle('mask_df.pickle')
