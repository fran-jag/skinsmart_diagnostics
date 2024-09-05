import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import subprocess

"""
This script creates a GUI application for uploading images and running predictions.
It allows users to select an image, input metadata, choose a prediction option,
and view the results of the prediction.
"""

def Run(selected_option):
    """
    Placeholder function for running predictions.
    
    Args:
        selected_option (str): The prediction option selected by the user.
    """
    print(f"Running prediction with selected option: {selected_option}")

def upload_image():
    """
    Function to handle image upload and display.
    
    This function opens a file dialog for the user to select an image,
    resizes the image to fit the display area, and updates the GUI accordingly.
    """
    def select_file():
        """
        Opens a file dialog and returns the selected file path.
        
        Returns:
            str: Path of the selected file, or None if no file was selected.
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
       
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
       
        return file_path
    
    # Usage
    file_path = select_file()
    if file_path:
        try:
            img = Image.open(file_path)
            # Get the original dimensions of the image
            original_width, original_height = img.size
            # Define the maximum dimensions available for the image
            max_width, max_height = 600, 450
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = min(max_width / original_width, max_height / original_height)
            # Calculate the new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            # Resize the image to the calculated dimensions
            img = img.resize((new_width, new_height), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            panel.config(image=img, text='')
            panel.image = img  # Keep a reference to avoid garbage collection
            # Store the file path
            global current_image_path
            current_image_path = file_path
            
            # Adjust window size based on the new image dimensions
            new_window_width = max(500, new_width + 50)  # Add some padding
            new_window_height = new_height + 530  # Add space for other widgets
            root.geometry(f"{new_window_width}x{new_window_height}")
            
            # Center the window on the screen
            root.update_idletasks()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x = (screen_width - new_window_width) // 2
            y = (screen_height - new_window_height) // 2
            root.geometry(f"+{x}+{y}")
            
        except Exception as e:
            print(f"Error loading image: {e}")

def on_closing():
    """
    Function to handle the window close event.
    
    This function is called when the user attempts to close the application window.
    It performs necessary cleanup and exits the application.
    """
    print("Window is closing. Cleaning up and exiting...")
    root.quit()  # Stop the mainloop
    root.destroy()  # Destroy the window

def predict():
    """
    Function to handle the prediction process.
    
    This function is called when the user clicks the 'Predict' button.
    It runs the prediction script with the selected image and metadata,
    and displays the results in the GUI.
    """
    if current_image_path and selected_option.get() != "Select an option":
        try:
            # Path to your prediction script
            script_path = "./src/app/predict.py"
            
            # Prepare metadata
            metadata = f"{sex_var.get()},{age_var.get()},{localization_var.get()}"
            
            # Run the prediction script with the image path, selected option, and metadata as arguments
            result = subprocess.run(["python", script_path, current_image_path, selected_option.get(), metadata], 
                                    capture_output=True, text=True, check=True)
            
            # Display the output
            output_label.config(text=f"Prediction Result:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            output_label.config(text=f"Error in prediction script:\n{e.stderr}")
    else:
        output_label.config(text="Please upload an image and select an option first.")

# Create the main application window
root = tk.Tk()
root.title("Image Upload and Prediction")
root.geometry("500x600")  # Increased height to accommodate new fields

# Bind the closing event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Image panel for displaying the uploaded image
panel = tk.Label(root, text="No image uploaded", bg="lightgray")
panel.pack(pady=20)

# Upload Button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=5)

# OptionMenu (Dropdown) setup
selected_option = tk.StringVar()
selected_option.set("Select an option")  
options = ["Unet -> CNN -> RF", "Other"]
dropdown = tk.OptionMenu(root, selected_option, *options)
dropdown.pack(pady=5)

# Metadata input fields
metadata_frame = ttk.LabelFrame(root, text="Metadata")
metadata_frame.pack(pady=10, padx=10, fill="x")

# Sex input
sex_var = tk.StringVar()
ttk.Label(metadata_frame, text="Sex:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
sex_combobox = ttk.Combobox(metadata_frame, textvariable=sex_var, values=["male", "female", "unknown"])
sex_combobox.grid(row=0, column=1, padx=5, pady=5)

# Age input
age_var = tk.StringVar()
ttk.Label(metadata_frame, text="Age:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
age_entry = ttk.Entry(metadata_frame, textvariable=age_var)
age_entry.grid(row=1, column=1, padx=5, pady=5)

# Localization input
localization_var = tk.StringVar()
ttk.Label(metadata_frame, text="Localization:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
localization_entry = ttk.Combobox(metadata_frame, textvariable=localization_var, values=['abdomen', 'acral', 'back', 'chest', 
                                     'ear', 'face', 'foot', 'genital', 
                                     'hand', 'lower extremity', 'neck', 
                                     'scalp', 'trunk', 'unknown', 'upper extremity'])
localization_entry.grid(row=2, column=1, padx=5, pady=5)

# Global variable to store the current image path
current_image_path = None

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=5)

# Label to display prediction output
output_label = tk.Label(root, text="", wraplength=440, justify="left", font=('arial', 13))
output_label.pack(pady=10)

# Run the application
if __name__ == "__main__":
    root.mainloop()