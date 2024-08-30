"""
Creates and organizes a file structure for tf.keras model training based on image metadata.

This function reads a metadata CSV file containing image information and reorganizes
the image files into a structured directory layout suitable for tf.keras model training.
It creates subdirectories for each unique label (diagnosis) and moves the corresponding
images into these subdirectories.

The final model structure is:
data/
├─ label_1/
│  ├─ image_1.jpg
│  ├─ image_2.jpg
│  ├─  ...
├─ label_2/
│  ├─ image_1.jpg
│  ├─ image_2.jpg
│  ├─  ...
├─ ...

Function workflow:
1. Prompts user to input the path to the metadata CSV file.
2. Reads the metadata file using pandas.
3. Prompts user to input the directory containing the image files.
4. Constructs source and destination paths for each image.
5. Creates subdirectories for each unique label (dx) in the metadata.
6. Prints the number of files per label.
7. Moves each image to its corresponding label subdirectory.

Parameters:
    None

Returns:
    None

Required CSV columns:
    - image_id: Unique identifier for each image (without file extension)
    - dx: Diagnosis or label for each image

User Inputs:
    - metadata_dir: Full path to the metadata CSV file
    - data_dir: Directory containing the image files

Notes:
    - Ensure that the metadata CSV file has 'image_id' and 'dx' columns.
    - All image files should be in .jpg format.
    - The function assumes all images are initially in the same directory.
    - Existing subdirectories with label names will not be overwritten.
    - This function modifies the file system, moving files from their original location.

Example usage:
    move_images()

    # User will be prompted to enter:
    # 1. Path to metadata CSV file
    # 2. Path to directory containing images

    # Function will then process and move the images, providing progress updates.

Raises:
    FileNotFoundError: If the metadata file or image directory doesn't exist.
    KeyError: If required columns are missing from the metadata CSV.
    PermissionError: If the function lacks permission to create directories or move files.
"""

import os
import pathlib
import pandas as pd

def move_images():
    metadata_dir = input("Please select the metadata file. \n")
    metadata_df = pd.read_csv(metadata_dir)



    data_dir = input("Please enter your data directory. \n")
    if data_dir[-1] != "/":
        data_dir = data_dir + "/"

    metadata_df['source'] = data_dir + metadata_df['image_id'] + '.jpg'
    metadata_df['destination'] = data_dir + metadata_df['dx'] + "/" + metadata_df['image_id'] + '.jpg'

    print(metadata_df.loc[0,'source'])
    print(metadata_df.loc[0, 'destination'])


    for folder in metadata_df['dx'].unique():
        pathlib.Path(data_dir+folder).mkdir(exist_ok=True)
        print(f"Created {folder} folder.")


    print('The number of files per label is:')
    diagns = metadata_df.groupby('dx').count()['image_id']
    print(diagns)
    print("-"*10)


    print("Moving images...")
    for index in metadata_df.index:
        os.rename(metadata_df.loc[index, 'source'], metadata_df.loc[index, 'destination'])

    print("All images have been moved to their respective labels.")

if __name__ == "__main__":
    move_images()