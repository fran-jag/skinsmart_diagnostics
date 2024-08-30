## This GUI is implemented with tkinter and contains various functions for image selection, metadata input and the execution of predictions.

Main functions:

1. **Image upload and display:**
    - Users can upload an image via a file dialogue. The image is then displayed in the GUI, automatically scaled to fit the available display area.  
2 **Metadata input:**
    - There are input fields for gender, age and localisation. These fields are used to collect metadata for the prediction.  
3 **Prediction options:**
    - Users can select a prediction option from a drop-down menu (e.g. select models: ‘Unet -> CNN -> RF’).  
4. **Run prediction:**
    - After uploading the image and entering the metadata, the user can click on the ‘Predict’ button. An external script is then executed that performs the prediction based on the data provided. The result is displayed in the GUI.ng.  

### Technical details:

- **Tkinter:** Used for GUI creation.  
- **PIL (Pillow):** Used to open and scale the uploaded image.  
- **Subprocess:** Enables the execution of external Python scripts for prediction.  

### Structure of the GUI:

- **Label:** Display of the uploaded image.  
- **Button:** Upload button to upload an image.  
- **Dropdown menu:** Selection of the prediction option.  
- **Metadata input:** Fields for gender, age and localisation.  
- **Predict button:** Starts the prediction and displays the results.  