from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import models as m

cnn_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

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
test_pipeline.get_steps()

#Configure img and metadata
#Select image - ensure your image paths or change accordingly
img = load_img(current_image_path,
            target_size=(unet_model.unet_train_params._WIDTH, unet_model.unet_train_params._HEIGHT)
            )
img_array = img_to_array(img)
#Create metadata
metadata_test = m.Metadata(sample_id = 'HAM_0006416', age = 35, sex='female', localization='upper extremity')

pipeline_preds = test_pipeline.predict(img_array, metadata_test)
print('-'*20)
print("All data processed. Printing results...")
for i, prediction in enumerate(pipeline_preds):
    print(cnn_classes[i], f": {prediction*100:.1f}%")  