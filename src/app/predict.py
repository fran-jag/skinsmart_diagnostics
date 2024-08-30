import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys

image_path = sys.argv[1]
selected_option = sys.argv[2]
metadata = sys.argv[3].split(',')
sex, age, localization = metadata

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import models as m

tf.compat.v1.logging .set_verbosity(40)

cnn_classes = [
    "Actinic Keratosis / Intraepithelial Carcinoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi (Moles)",
    "Vascular Lesions"
]

#Initiate and load unet model
unet_model = m.unet()
unet_model.load_model('../../models/Unet_31M_mask.keras')

#Initiate and load CNN model
cnn_model = m.cnn()
cnn_model.load_model('../../models/CNN_97_diagnosis_segmented.keras')

#Initiate and load RF
rf_model = m.RandomForest()
rf_model.load_model('../../models/model_Random_Forest.pkl')

#Initiate pipeline
models = {'unet':unet_model, 'cnn':cnn_model, 'rf':rf_model}
types = ['img', 'image', 'metadata']
test_pipeline = m.Pipeline(models, types)

#Configure img and metadata
#Select image - ensure your image paths or change accordingly
img = load_img(image_path,
            target_size=(unet_model.unet_train_params._WIDTH, unet_model.unet_train_params._HEIGHT)
            )
img_array = img_to_array(img)
#Create metadata
metadata_test = m.Metadata(sample_id = '0', age = age, sex=sex, localization=localization)

pipeline_preds = test_pipeline.predict(img_array, metadata_test)
print('-'*20)
print("All data processed. Printing results...")
print('-'*20)

#Sort the labels and predictions
sorted_pairs = sorted(zip(pipeline_preds, cnn_classes), reverse=True)
preds_sorted, classes_sorted = zip(*sorted_pairs)

# Convert back to lists (zip returns tuples)
preds_sorted = list(preds_sorted)
classes_sorted = list(classes_sorted)

for i, prediction in enumerate(preds_sorted):
    print(classes_sorted[i], f": {prediction*100:.1f}%")  