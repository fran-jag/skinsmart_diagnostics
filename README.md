![SkinSmart banner](https://i.ibb.co/ZGmd0Z5/banner.png)

# SkinSmart Diagnostics

### Project Overview
Objective: To develop an AI-powered system for diagnosing skin cancer using image recognition neural networks.

Context: Manual visual examination of skin malformations is subjective and error-prone. Skin cancer is the most common cancer in Germany, with over 270,000 new cases annually, including more than 30,000 cases of malignant melanoma, the deadliest form.

Significance: Early detection of skin cancer is critical for better patient outcomes. This tool aids in identifying potentially cancerous lesions early, thus improving the accuracy of initial diagnoses at the GP level.

Goal: To classify skin lesions into 7 possible diagnoses, some cancerous, and enable selection and training of different AI models within the application.  
A presentation of our project can be seen [here](/presentation/skinsmart_presentation.pdf)


## Team Members

- **Marcel Sonnenschein** (MSc. Physics) [GitHub](https://github.com/MarcelSonne)

Roles: Scrum Master, Infrastructure Manager  
Tasks: Developing Project Infrastructure, Developing Database  

- **Mahtab Lashgari** (MSc. Neurobiology) [GitHub](https://github.com/MahtabLashgari)

Roles: Data Scientist, Frontend developer  
Tasks: EDA, Machine Learning on Metadata, GUI development  

- **Francisco J. Arriaza G.** (PhD. Biochemistry) [GitHub](https://github.com/fran-jag)

Roles: Quality Control, Deep Learning Expert  
Tasks: Develop Image Classification Models, Final Ensemble  

## Code

This project consists of several python scripts that serve different purposes located in /src/:

1. **/src/visual/main.py**: 

This is the main script containing the GUI elements and links
the user interface with the prediction scripts.

2. **/src/app/predict.py**: 

Main prediction script engine ran by main.py

3. **models.py**: 

File containing the Model classes and test for said models.

## Installation and Setup

To set up the project locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/MarcelSonne/Skin_cancer_diagnosis_tool.git
```
2. Navigate to the project directory:
```
cd your-repository
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Download the dataset, unzip images into ~/data/images. The original dataset can be acquired from the link: [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

5. Run the file data_path_preparation.py which will perpare the directory for Tensorflow Keras training based on the metadata provided in the dataset.

**Note:** If any of the above files are missing, the corresponding functionality may not work as expected.



## Dataset

The dataset consists of 10,015 dermatoscopic images with associated metadata. It was sourced from a Harvard study. More than 50% of lesions are confirmed through histopathology.

## Attribute Information

The dataset contains the following attributes:

1. Dermatoscopic images of skin lesions
2. Demographic information of each lesion (metadata)
    1. Location of the lesion
    2. Sex of the patient
    3. Age of the patient
    3. Diagnosis (7 possible classes)


## EDA/Cleaning
**~/notebooks/EDA_1.ipynb**

- Visualization of frequency distribution of different classes (diseases)
- Visualization of gender distribution of patients
- Analysis of disease location across gender
- Feature extraction and enhancement from images
- Removal of artifacts from images

## Model Choices

- Metadata Model: Random Forest
- Image Classification Model:
    1. Base 48M-parameters CNN model
    2. Optimized 97M-parameters CNN model
- Image Preprocessing: UNet Neural Network
- Final Ensemble Model: Combination of UNet preprocessing, CNN image classification, and Random Forest metadata predictions

## Results

Initial test ROC AUC = ~0.6, with the final model test AUC of 0.8. Independent data testing showed AUC of ~0.6 with the final model, meaning it lacks generalization abilities which can be optimized.

## Prediction Function

When a prediction is made:

1. The image is preprocessed using the UNet model
2. The preprocessed image is classified using the CNN model
3. Metadata is analyzed using the Random Forest model
4. The ensemble model combines these predictions to provide a final diagnosis

## Final Remarks

The SkinSmart Diagnostics project was done as part of the Portfolio Course from StackFuel in 2024.
