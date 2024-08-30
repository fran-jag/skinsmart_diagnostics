#Import os and set os warning level.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Import standard libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import joblib

#Import tensorflow and tf.Keras.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

#Import sklearn
from sklearn.preprocessing import LabelEncoder

#Import Abstract Base Class.
from abc import ABC, abstractmethod

#Import needed scripts.
import model_unet
import model_cnn

#Silence TF's warnings.
tf.get_logger().setLevel('INFO')

class Metadata:
    """
    A class to represent metadata for a sample.

    Attributes:
        sample_id (str): Unique identifier for the sample.
        age (int): Age of the patient.
        sex (str): Sex of the patient.
        localization (str): Localization of the sample on the body.
    """

    def __init__(self, sample_id, age, sex, localization):
        self.sample_id = sample_id
        self.age = age
        self.sex = sex
        self.localization = localization

class Model(ABC):
    """
    Abstract base class for all models.

    This class defines the interface for all model classes.
    """

    @abstractmethod
    def load_model(self):
        """Load a pre-trained model."""
        pass

    @abstractmethod
    def predict(self):
        """Make predictions using the model."""
        pass

    @abstractmethod
    def summary(self):
        """Provide a summary of the model."""
        pass

    # TODO - Define train method.
    # @abstractmethod
    # def train(self):
    #     pass

class unet(Model):
    """
    A class representing a U-Net model for image segmentation.

    This class encapsulates the functionality for loading, building, and using a U-Net model
    for predicting segmentation masks on images.

    Attributes:
        unet_train_params (model_unet.Parameters): Parameters for training the U-Net model.
        unet_ori_params (model_unet.Parameters): Original parameters for the U-Net model.
        model: The U-Net model instance.
    """

    def __init__(self):
        """Initialize the unet class."""
        # Initialize parameters for training and original model
        self.unet_train_params = model_unet.Parameters()
        self.unet_ori_params = model_unet.Parameters(600, 450)
                     
        # Build the initial model
        self.model = model_unet.buil_model(self.unet_train_params)
    
    def load_model(self, model_dir):
        """
        Load a pre-trained U-Net model.

        Args:
            model_dir (str): Directory path for the model file.
        """
        self.model = tf.keras.models.load_model(model_dir)

    def build_model(self, model_weights):
        """
        Build and load the U-Net model with pre-trained weights.

        Args:
            model_weights (str): Path to the pre-trained weights.
        """
        if model_weights:
            model_unet.load_train_model(self.model, self.unet_train_params, trained=True, model_weights=model_weights)

    def summary(self):
        """
        Get a summary of the U-Net model architecture.

        Returns:
            str: A string summary of the model architecture.
        """
        return self.model.summary()

    def predict(self, array=None, image=None):
        """
        Predict a segmentation mask for the given input.

        This method can accept either a numpy array or a PIL Image as input.

        Args:
            array (numpy.ndarray, optional): Input image as a numpy array.
            image (PIL.Image, optional): Input image as a PIL Image object.

        Returns:
            numpy.ndarray: The processed image with the segmentation mask applied.

        Raises:
            ValueError: If neither array nor image is provided.
        """
        if image:
            # Convert PIL Image to numpy array
            img_array = img_to_array(image)
            # Expand dimensions to create a batch of size (1, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)
        elif array is not None and array.any():
            # Expand dimensions for the input array
            img_array = np.expand_dims(array, axis=0)
        else:
            raise ValueError("Please provide either a numpy array or a PIL Image.")

        # Preprocess the image by scaling to [0,1]
        img_array = img_array / 255.0

        # Make prediction using the model
        prediction = self.model.predict(img_array, verbose=0)

        # Reshape the output and apply threshold
        reshaped_prediction = prediction.squeeze()
        mask = reshaped_prediction > 0.5

        # Convert mask to PIL Image
        img_pil = PIL.Image.fromarray(np.asarray(PIL.Image.fromarray(mask, 'L'))*255, 'L')
        
        # Resize mask to original dimensions
        final_mask = img_pil.resize((self.unet_ori_params._WIDTH, self.unet_ori_params._HEIGHT))

        if image:
            processed_image = PIL.Image.composite(image, final_mask.convert('RGB'), final_mask.convert('L'))
        if array is not None and array.any():
            ori_img = tf.keras.utils.array_to_img(array).resize((self.unet_ori_params._WIDTH, self.unet_ori_params._HEIGHT))
            processed_image = PIL.Image.composite(ori_img, final_mask.convert('RGB'), final_mask.convert('L'))
        
        return np.expand_dims(processed_image, axis=0)

class cnn(Model):
    """
    A class representing a CNN model for image classification.

    Attributes:
        model: The CNN model instance.
        labels (list): List of class labels.
    """

    def __init__(self):
        """Initialize the cnn class."""
        self.model = []
        self.labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    def load_model(self, model_dir):
        """
        Load a pre-trained CNN model.

        Args:
            model_dir (str): Directory path for the model file.
        """
        self.model = tf.keras.models.load_model(model_dir)
    
    def build_model(self, model_weights):
        """
        Build the CNN model with pre-trained weights.

        Args:
            model_weights (str): Path to the pre-trained weights.
        """
        if model_weights:
            pass  # TODO: Implement weight loading logic

    def predict(self, image):
        """
        Make predictions using the CNN model.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Prediction probabilities for each class.
        """
        return self.model(image, training=False).numpy().flatten()

    def summary(self):
        """
        Get a summary of the CNN model architecture.

        Returns:
            str: A string summary of the model architecture.
        """
        return self.model.summary()

class RandomForest(Model):
    """
    A class representing a Random Forest model for classification based on metadata.

    Attributes:
        model: The Random Forest model instance.
        classes (list): List of class labels.
        sex_encoder (list): List of sex categories for encoding.
        localization_encoder (list): List of localization categories for encoding.
    """

    def __init__(self):
        """Initialize the RandomForest class."""
        self.model = []
        self.classes = ['nv', 'df', 'vasc', 'bkl', 'bcc', 'akiec', 'mel']
        self.sex_encoder = ['female', 'male', 'unknown']
        self.localization_encoder = ['abdomen', 'acral', 'back', 'chest', 
                                     'ear', 'face', 'foot', 'genital', 
                                     'hand', 'lower extremity', 'neck', 
                                     'scalp', 'trunk', 'unknown', 'upper extremity']

    def load_model(self, rf_path):
        """
        Load a pre-trained Random Forest model.

        Args:
            rf_path (str): Path to the saved Random Forest model file.
        """
        self.model = joblib.load(rf_path)

    def predict(self, metadata: Metadata):
        """
        Make predictions using the Random Forest model based on metadata.

        Args:
            metadata (Metadata): Metadata object containing patient information.

        Returns:
            numpy.ndarray: Reordered prediction probabilities for each class.
        """
        # Define RF to CNN mapping
        cnn_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        mapping = {class_B: cnn_classes.index(class_B) for class_B in self.classes}
        
        # Get metadata from class
        age = metadata.age
        sex = self.sex_encoder.index(metadata.sex)
        localization = self.localization_encoder.index(metadata.localization)
        
        # Predict and rearrange
        predict_df = pd.DataFrame({'age': age, 'sex' : sex, 'localization': localization}, index=[0])
        predictions = self.model.predict_proba(predict_df).flatten()
        
        reordered_predictions = np.zeros_like(predictions)
        for i, prediction in enumerate(predictions):
            new_index = mapping[self.classes[i]]
            reordered_predictions[new_index] = prediction

        return reordered_predictions

    def summary(self):
        """
        Print a summary of the trained RandomForestClassifier.

        Prints:
        - Number of trees
        - Max depth
        - Min samples split
        - Min samples leaf
        - Feature importances (top 10 or all if less than 10)
        - Out-of-bag score (if available)
        """
        print("Random Forest Classifier Summary")
        print("================================")
        
        # Basic parameters
        print(f"Number of trees: {self.model.n_estimators}")
        print(f"Max depth: {self.model.max_depth}")
        print(f"Min samples split: {self.model.min_samples_split}")
        print(f"Min samples leaf: {self.model.min_samples_leaf}")
        
        # Feature importances
        importances = self.model.feature_importances_
        feature_names = getattr(self.model, 'feature_names_in_', [f'Feature {i}' for i in range(len(importances))])
        
        sorted_idx = np.argsort(importances)
        top_k = min(10, len(importances))
        top_features = sorted_idx[-top_k:][::-1]
        
        print("\nTop Feature Importances:")
        for idx in top_features:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        
        # Out-of-bag score
        if hasattr(self.model, 'oob_score_'):
            print(f"\nOut-of-bag score: {self.model.oob_score_:.4f}")
        else:
            print("\nOut-of-bag score not available (oob_score was set to False during training)")

class Pipeline():

    def __init__(self, models: dict = None, types: list = None):
        output_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.models = models
        self.keys = list(models.keys())
        self.steps = list(models.values())
        self.types = types
        self.img_index = [i for i, x in enumerate(self.types) if x == 'img' or x == 'image']
        self.metadata_index = [i for i, x in enumerate(self.types) if x == 'metadata']

    def get_steps(self):
        print("Image steps:")
        for i in self.img_index:
            print(f"{self.keys[i]}: {self.steps[i]}")
        
        print("Metadata steps:")
        for i in self.metadata_index:
            print(f"{self.keys[i]}: {self.steps[i]}")

    def predict(self, input, metadata: Metadata):
        def image_steps(input):
            print("Processing images...")
            for i in self.img_index:
                out = self.steps[i].predict(input)
                input = out
            return input
        img_probabs = image_steps(input)

        def metadata_steps(metadata):
            print("Processing metadata...")
            for j in self. metadata_index:
                out = self.steps[j].predict(metadata)
            return out
        metadata_probabs = metadata_steps(metadata)

        averaged_probabs = [np.average([img_probabs[x], metadata_probabs[x]], weights=[0.6, 0.4]) for x in range(len(img_probabs))]

        return averaged_probabs

#Test classes
if __name__ == '__main__':

    #Change dir to current path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #Select test
    selector = input("Select test: 'image', 'metadata', 'both' or 'pipeline' \n")

    def test_image():
        #Initiate and load unet model
        unet_model = unet()
        unet_model.load_model('../../models/Unet_31M_mask.keras')
        
        #Select image - ensure your image paths or change accordingly
        img_src = '../../data/images/ISIC_0028137.jpg'
        img = load_img(img_src,
                    target_size=(unet_model.unet_train_params._WIDTH, unet_model.unet_train_params._HEIGHT)
                    )
        img_array = img_to_array(img)
        
        #Predict mask
        test_mask = unet_model.predict(array=img_array)
        
        #Initiate and load CNN model
        cnn_model = cnn()
        cnn_model.load_model('../../models/CNN_97_diagnosis_segmented.keras')
        
        #Predict class
        predictions = cnn_model.predict(test_mask)
        
        #Print probabilities
        for i, prediction in enumerate(predictions):
            print(cnn_model.labels[i] + f":  {prediction*100:.1f}%")
    
    def test_metadata():
        cnn_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        rf_model = RandomForest()
        rf_model.load_model('../../models/model_Random_Forest.pkl')
        metadata_test = Metadata(sample_id = 'HAM_0006416', age = 35, sex='female', localization='upper extremity')
        rf_predictions = rf_model.predict(metadata_test)
        for i, prediction in enumerate(rf_predictions):
            print(cnn_classes[i], f": {prediction*100:.1f}%")
    
    if selector == 'image':
        test_image()

    elif selector == 'metadata':
        test_metadata()

    elif selector == 'both':
        print("Testing image model...", end='\n')
        test_image()
        print("Testing metadata model...", end='\n')
        test_metadata()
    
    elif selector == 'pipeline':
        cnn_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        #Initiate and load unet model
        unet_model = unet()
        unet_model.load_model('../../models/Unet_31M_mask.keras')

        #Initiate and load CNN model
        cnn_model = cnn()
        cnn_model.load_model('../../models/CNN_97_diagnosis_segmented.keras')
        
        #Initiate and load RF
        rf_model = RandomForest()
        rf_model.load_model('../../models/model_Random_Forest.pkl')

        #Initiate pipeline
        models = {'unet':unet_model, 'cnn':cnn_model, 'rf':rf_model}
        types = ['img', 'image', 'metadata']
        test_pipeline = Pipeline(models, types)
        test_pipeline.get_steps()

        #Configure img and metadata
        #Select image - ensure your image paths or change accordingly
        img_src = '../../data/images/ISIC_0028137.jpg'
        img = load_img(img_src,
                    target_size=(unet_model.unet_train_params._WIDTH, unet_model.unet_train_params._HEIGHT)
                    )
        img_array = img_to_array(img)
        #Create metadata
        metadata_test = Metadata(sample_id = 'HAM_0006416', age = 35, sex='female', localization='upper extremity')

        pipeline_preds = test_pipeline.predict(img_array, metadata_test)
        print('-'*20)
        print("All data processed. Printing results...")
        for i, prediction in enumerate(pipeline_preds):
            print(cnn_classes[i], f": {prediction*100:.1f}%")        
        