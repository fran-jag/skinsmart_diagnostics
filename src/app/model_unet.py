'''
Adapted from
https://github.com/ZeeTsing/Carvana_challenge/blob/master/3_Unet_trained.ipynb
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import keras
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import os
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K



import pathlib

def base_setup(ratio: float = 0.1):
    data_dir = input("Please enter your data directory. \n")
    mask_dir = input("Please enter your mask/segments directory. \n")

    all_images = os.listdir(data_dir)

    to_train = ratio # ratio of number of train set images to use
    total_train_images = all_images[:int(len(all_images)*to_train)]

    print("{} images found in data directory.".format(total_train_images))

    return data_dir, mask_dir, total_train_images

class Parameters():
    """
    A class to encapsulate and manage parameters used for image processing and model training/testing.
    
    Attributes:
        _WIDTH (int): The width of the input images. Default is 448.
        _HEIGHT (int): The height of the input images. Default is 448.
        _BATCH_SIZE (int): The number of samples per batch to load. Default is 2.

    Usage:
        params = Parameters(width=224, height=224, batch_size=2)
        image_width = params._WIDTH
        batch_size = params._BATCH_SIZE

    Note:
        The Unet arquitecture needs symetrical images with dimensions divisible by 64.
        The images will be rescaled to 448x448 by default.
    """
    def __init__(self, width: int = 448, height: int = 448, batch_size: int = 2):
        self._WIDTH = width 
        self._HEIGHT = height 
        self._BATCH_SIZE = batch_size

# split train set and test set
def get_train_test_split(total_train_images):
    train_images, validation_images = train_test_split(total_train_images, train_size=0.8, test_size=0.2,random_state = 0)
    return train_images, validation_images

# generator that we will use to read the data from the directory
def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
        """
        data_dir: where the actual images are kept
        mask_dir: where the actual masks are kept
        images: the filenames of the images we want to generate batches from
        batch_size: self explanatory
        dims: the dimensions in which we want to rescale our images, tuple
        """
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                # images
                original_img = load_img(data_dir + images[i])
                resized_img = original_img.resize(dims)
                array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
                
                # masks
                original_mask = load_img(mask_dir + images[i].split(".")[0] + '_segmentation.png')
                resized_mask = original_mask.resize(dims)
                array_mask = img_to_array(resized_mask)/255
                labels.append(array_mask[:, :, 0])

            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels.reshape(-1, dims[0], dims[1], 1)

# generator that we will use to read the data from the directory with random augmentation
def data_gen_aug(data_dir, mask_dir, images, batch_size, dims):
        """
        data_dir: where the actual images are kept
        mask_dir: where the actual masks are kept
        images: the filenames of the images we want to generate batches from
        batch_size: self explanatory
        dims: the dimensions in which we want to rescale our images, tuple
        """
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                # read images and masks
                original_img = load_img(data_dir + images[i])
                original_mask = load_img(mask_dir + images[i].split(".")[0] + '_segmentation.png')
                
                # transform into ideal sizes
                resized_img = original_img.resize(dims)
                resized_mask = original_mask.resize(dims)
              
                # add random augmentation > here we only flip horizontally
                if np.random.random() < 0.5:
                  resized_img = resized_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                  resized_mask = resized_mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

                array_img = img_to_array(resized_img)/255
                array_mask = img_to_array(resized_mask)/255

                imgs.append(array_img)
                labels.append(array_mask[:, :, 0])
                
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels.reshape(-1, dims[0], dims[1], 1)

#generator for train and validation data set
def generate_dataset(data_dir, mask_dir, train_images, validation_images, BATCH_SIZE, WIDTH, HEIGHT):
    train_gen = data_gen_aug(data_dir, mask_dir, train_images, BATCH_SIZE, (WIDTH, HEIGHT))
    val_gen = data_gen_small(data_dir, mask_dir, validation_images, BATCH_SIZE, (WIDTH, HEIGHT))
    return train_gen, val_gen

def buil_model(params: Parameters):
    WIDTH = params._WIDTH
    HEIGHT = params._HEIGHT

    #Set up uNet model
    def down(input_layer, filters, pool=True):
        conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
        residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
        if pool:
            max_pool = MaxPool2D()(residual)
            return max_pool, residual
        else:
            return residual

    def up(input_layer, residual, filters):
        filters=int(filters)
        upsample = UpSampling2D()(input_layer)
        upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
        concat = Concatenate(axis=3)([residual, upconv])
        conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
        conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
        return conv2  

    # Make a custom U-nets implementation.
    filters = 64
    input_layer = Input(shape = [WIDTH, HEIGHT, 3])
    layers = [input_layer]
    residuals = []

    # Down 1
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)

    filters *= 2

    # Down 2
    d2, res2 = down(d1, filters)
    residuals.append(res2)

    filters *= 2

    # Down 3
    d3, res3 = down(d2, filters)
    residuals.append(res3)

    filters *= 2

    # Down 4
    d4, res4 = down(d3, filters)
    residuals.append(res4)

    filters *= 2

    # Down 5
    d5 = down(d4, filters, pool=False)

    # Up 1
    up1 = up(d5, residual=residuals[-1], filters=filters/2)
    filters /= 2

    # Up 2
    up2 = up(up1, residual=residuals[-2], filters=filters/2)

    filters /= 2

    # Up 3
    up3 = up(up2, residual=residuals[-3], filters=filters/2)

    filters /= 2

    # Up 4
    up4 = up(up3, residual=residuals[-4], filters=filters/2)

    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

    model = Model(input_layer, out)

    # Now let's use Tensorflow to write our own dice_coeficcient metric, 
    # which is a effective indicator of how much two sets overlap with each other
    @keras.saving.register_keras_serializable()
    def dice_coef(y_true, y_pred):
        smooth = 1.
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer = adam, loss = BinaryCrossentropy(), metrics=['accuracy', dice_coef])
    
    return model


def load_train_model(model, params: Parameters, train_gen = None, val_gen = None, save_dir = None, trained: bool = True, model_weights: str = None):
    if trained:
        model.load_weights(model_weights)
        return "Loaded model weights \n {}".format(model_weights)
    
    else:
        BATCH_SIZE = params._BATCH_SIZE

        checkpoint_path = save_dir + "cp-{epoch:04d}.weights.h5"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Create a callback that saves the model's weights every epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir, 
            verbose=1, 
            save_weights_only=True,
            save_freq='epoch')

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                restore_best_weights=False
                                                )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.2,
                                    patience=3,
                                    verbose=1,
                                    min_delta=1e-3,min_lr = 1e-6
                                    )

        history = model.fit(train_gen, callbacks=[cp_callback,early_stop,reduce_lr],
                    steps_per_epoch=int(np.ceil(float(len(train_images)) / float(BATCH_SIZE))),
                    epochs=100,
                    validation_steps=int(np.ceil(float(len(validation_images)) / float(BATCH_SIZE))),
                    validation_data = val_gen)
        
        return history
    
def predict_single_image(model, image_path, train_params: Parameters, original_params: Parameters):
    target_size = (train_params._WIDTH, train_params._HEIGHT)
    
    # Load the image
    img = load_img(image_path, target_size=target_size)
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Expand dimensions to create a batch of size (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image (scaling to [0,1])
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Reshape the output to (448, 448)
    reshaped_prediction = prediction.squeeze()
    
    mask = reshaped_prediction > 0.3

    img_pil = PIL.Image.fromarray(np.asarray(PIL.Image.fromarray(mask, 'L'))*255, 'L')
    
    final_mask = img_pil.resize((original_params._WIDTH, original_params._HEIGHT))
    return final_mask
     

def predict_batch(model, folder_path, output_path, params: Parameters, original_params: Parameters):

    images_list = os.listdir(folder_path)

    def load_images_from_folder(folder_path, params: Parameters, label=0):
        """
        Load all images from a single folder and assign them the same label.
        
        Args:
        folder_path (str): Path to the folder containing images.
        width (int): Target width for the images.
        height (int): Target height for the images.
        batch_size (int): Number of images to load in each batch.
        label (int, optional): Label to assign to all images. Defaults to 0.
        
        Returns:
        tuple: (ImageDataGenerator, number of samples)
        """
        width = params._WIDTH
        height = params._HEIGHT
        batch_size = params._BATCH_SIZE

        # Get all image files from the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create a DataFrame with file paths and labels
        df = pd.DataFrame({
            'filename': image_files,
            'label': [label] * len(image_files)
        })
        
        # Create ImageDataGenerator
        datagen = ImageDataGenerator(rescale=1./255)
        
        # Create the generator
        image_generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=folder_path,
            x_col='filename',
            y_col='label',
            target_size=(width, height),
            color_mode='rgb',
            class_mode='raw',  # 'raw' for single label
            batch_size=batch_size,
            shuffle=False
        )
        
        return image_generator, len(image_files)
    
    train_date_gen, nb_samples = load_images_from_folder(folder_path, params)
    
    predict = model.predict(train_date_gen,
                            steps = int(np.ceil(nb_samples/params._BATCH_SIZE)))
    
    def get_predicted_img(output, DIM = (params._WIDTH, params._HEIGHT)):
        data = np.reshape(output, DIM)
        mask = data > 0.4
        return np.asarray(PIL.Image.fromarray(mask, 'L'))
    
    def get_img_resize(fname, DIM = (original_params._WIDTH, original_params._HEIGHT)):
        img_pil = PIL.Image.fromarray(fname*255, 'L')
        mask = img_pil.resize(DIM)
        return mask
     
    for i, image in enumerate(predict):
        print("{} processed.".format(images_list[i]))
        images = get_img_resize(get_predicted_img(image))
        images.save(output_path+images_list[i])
        images.close()