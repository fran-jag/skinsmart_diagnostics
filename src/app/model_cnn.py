"""
Skin Lessions Classification Program

This program provides a set of tools and functions for building, training, and evaluating
an image classification model using TensorFlow and Keras. It is designed to work with
directories of images organized by class labels.

Main Components:
1. Utility Functions:
   - get_version(): Prints TensorFlow version and available GPUs.
   - choose_directory(): Prompts user for input directory.
   - get_labels(): Counts and prints labels in a directory.
   - files_count(): Counts and prints total images and labels.

2. Parameters Class:
   - Stores batch size, image height, and width for processing.

3. Data Preparation:
   - train_test_split(): Splits image data into training and validation sets.

4. Model Creation and Training:
   - create_and_compile(): Creates and compiles a CNN model.
   - train_model(): Trains the model on provided datasets.

5. Visualization and Reporting:
   - model_summary(): Prints model architecture summary.
   - plot_training(): Plots training and validation metrics.

6. Main Execution:
   - _secret_test(): Demonstrates the full workflow of the program.

Usage:
1. Ensure your image data is organized in directories by class labels.
2. Run the program and use the choose_directory() function to select your data.
    2.1. See data_path_preparation.py for a related script.
3. Use the Parameters class to set batch size and image dimensions.
4. Split your data, create and compile the model, then train it.
5. Visualize results using the provided plotting function.

Note: This program assumes a binary classification task by default, but can be 
adjusted for multi-class problems by modifying the create_and_compile() function.

Requirements:
- TensorFlow
- NumPy
- PIL
- Matplotlib

Example:
    base_dir = choose_directory()
    files_count(base_dir)
    params = Parameters(32, 450, 600)
    train_ds, val_ds = train_test_split(base_dir, params)
    model = create_and_compile()
    history = train_model(model, train_ds, val_ds)
    model_summary(model)
    plot_training(history)
"""


import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import pathlib

import PIL
import PIL.Image

import tensorflow as tf
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt



def get_version():
    '''Print current TF version and GPUs available'''
    print(f"TensorFlow version: {tf.__version__}")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def choose_directory():
    esc_count = 0
    while True:
        directory = input("Please enter the full path of the directory: ").strip()
        if directory[-1] != "/":
            directory = directory + "/"
        
        # Expand user directory if ~ is used
        directory = os.path.expanduser(directory)
        base_dir = pathlib.Path(directory)
        
        if os.path.isdir(directory):
            return base_dir
        elif esc_count >= 3:
            print("Exiting...\nPlease check directory and retry function.")
            return "invalid"
        else:
            print("Invalid directory. Please try again.")
            esc_count += 1

def get_labels(folder: pathlib.PosixPath):
    no_classes = len(os.listdir(folder))
    print("There are {} labels.".format(no_classes))
    print(os.listdir(folder))
    return no_classes

def files_count(x : pathlib.PosixPath):
    assert type(x) == pathlib.PosixPath, "Please check path."
    image_list = list(x.glob('*/*.jpg'))
    print("{} total images found.".format(len(image_list)))
    print("{} labels:".format(len(os.listdir(x))))
    for x in os.listdir(x):
        print(" "*5, x)


class Parameters():
    """
    A class to store and manage parameters for image processing or neural network operations.

    Attributes:
        _batch_size (int): The number of samples in each batch.
        _height (int): The height of the image or feature map.
        _width (int): The width of the image or feature map.

    Args:
        size (int): The batch size to be set.
        height (int): The height to be set.
        width (int): The width to be set.
    """
    def __init__(self, size: int, height: int, width: int):
        self._batch_size = size
        self._height = height
        self._width = width

def train_test_split(base_dir: pathlib.Path, parameters: Parameters, split: int = 0.2, seed: int = 0):
    """
    Split a directory of images into training and validation datasets.

    This function uses TensorFlow's image_dataset_from_directory utility to create
    dataset objects for training and validation. It splits the data in the given
    directory into 80% training and 20% validation sets.

    Args:
        base_dir (pathlib.Path): The base directory containing the image data.
            This directory should have subdirectories, each representing a class.
        parameters (Parameters): An instance of the Parameters class containing
            batch size, image height, and image width.
        split (int, optional): The fraction of data used for validation.
            Defaults to 0.2.
        seed (int, optional): Random seed for shuffling and transformations.
            Defaults to 0.

    Returns:
        tuple: A tuple containing two elements:
            - train_ds (tf.data.Dataset): The training dataset.
            - val_ds (tf.data.Dataset): The validation dataset.

    The returned datasets are configured with the following properties:
        - Image size is set according to the parameters.
        - Batch size is set according to the parameters.
        - The data is split with 80% for training and 20% for validation.

    Note:
        This function assumes that the base_dir has a structure where each subdirectory
        represents a different class, and contains images of that class.

    Example:
        >>> params = Parameters(batch_size=32, height=224, width=224)
        >>> train_ds, val_ds = train_test_split('/path/to/image/directory', params)
    """
    img_height = parameters._height
    img_width = parameters._width
    batch_size = parameters._batch_size

    train_ds = tf.keras.utils.image_dataset_from_directory(
                base_dir,
                validation_split = split,
                subset = 'training',
                seed = seed,
                image_size = (img_height, img_width),
                batch_size = batch_size
                )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
                base_dir,
                validation_split = split,
                subset="validation",
                seed = seed,
                image_size=(img_height, img_width),
                batch_size=batch_size
                )

    return train_ds, test_ds

def create_and_compile(classes: int):
    """
    Creates and compiles a Sequential CNN model for image classification.

    This function builds a Convolutional Neural Network (CNN) using TensorFlow's Keras API.
    The model is designed for binary classification by default, but can be adjusted for
    multi-class classification.

    Args:
        classes (int, optional): The number of classes for classification. Defaults to 2.

    Returns:
        tf.keras.Sequential: A compiled Keras Sequential model ready for training.

    Model Architecture:
        - Input: Implicitly defined by the first layer (expected to be image data)
        - Rescaling layer to normalize pixel values
        - 3 Convolutional layers with MaxPooling
        - Flatten layer
        - 2 Dense layers
        - Output Dense layer with sigmoid activation for binary classification

    The model is compiled with:
        - Adam optimizer
        - Binary Cross-Entropy loss
        - AUC (Area Under the Curve) metric

    Note:
        - The model assumes input images are in the range [0, 255] and rescales them to [0, 1].
        - The final layer uses sigmoid activation, suitable for binary classification.
        - For multi-class problems (classes > 2), consider changing the final layer and loss function.

    Example:
        >>> model = create_and_compile()
        >>> model.summary()
    """
    classes = classes

    if classes == 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])
              
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.AUC()]
                    )
    
    if classes > 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(classes, activation='softmax')
            ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=classes)]
                    )
    return model

def train_model(model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int = 100):
    """
    Train a Keras model using provided training and validation datasets.

    This function takes a compiled Keras model and trains it on the given datasets.
    It uses a fixed number of epochs and returns the training history.

    Args:
        model (tf.keras.Sequential): A compiled Keras Sequential model ready for training.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.

    Returns:
        tf.keras.callbacks.History: A History object. Its History.history attribute is
        a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values (if applicable).

    Note:
        - The function uses a fixed number of 8 epochs. Adjust this if needed for your specific use case.
        - No additional callbacks or custom training parameters are used in this basic implementation.
        - Ensure that the model is already compiled before passing it to this function.

    Example:
        >>> model = create_and_compile()
        >>> train_ds, val_ds = load_datasets()
        >>> history = train_model(model, train_ds, val_ds)
        >>> plt.plot(history.history['loss'], label='training loss')
        >>> plt.plot(history.history['val_loss'], label='validation loss')
        >>> plt.legend()
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                min_delta=0.02,
                patience=10, 
                restore_best_weights=True
                )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.1, 
                patience=5, 
                min_lr=1e-6
                )

    history = model.fit(train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[early_stopping, lr_scheduler]
                )
    return history
    
def model_sumary(model):
    """Print model summary"""
    print(model.summary())

def plot_trainig(history):
    
    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(history.history['auc'], label='auc')
    axs[0].plot(history.history['val_auc'], label='validation auc')
    axs[0].set_ylabel('auc')
    axs[0].legend(['training', 'validation'])

    axs[1].plot(history.history['loss'], label='loss')
    axs[1].plot(history.history['val_loss'], label='validation loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['training', 'validation'])


#Remove before publishing
def _secret_test():
    base_dir = choose_directory()
    files_count(base_dir)
    params = Parameters(32, 450, 600)
    train_ds, val_ds = train_test_split(base_dir, params)
    model = create_and_compile()
    history = train_model(model, train_ds, val_ds, 3)
    model_sumary(model)
    plot_trainig(history)
