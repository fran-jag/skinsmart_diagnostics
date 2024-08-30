import os
from PIL import Image

def create_segmented_images():
    """
    Create segmented images by combining original images with their corresponding mask images.

    This function prompts the user for input and output directories, then processes each image
    in the input directory by applying its corresponding mask (if available) and saves the
    result in the output directory.

    The function expects:
    - Original images in .jpg format
    - Mask images in .png format with '_segmentation' appended to the original filename

    User Inputs:
    - Image directory: Path to the directory containing original images
    - Mask directory: Path to the directory containing mask/segmentation images
    - Save directory: Path to the directory where processed images will be saved

    Note:
    - The function will add a trailing slash to directory paths if not provided.
    - For certain masks, numpy may reverse the masks.
    """
    # Set up image directories and save path
    image_dir = input("Please select your image directory.\n")
    mask_dir = input("Please select your mask or segmentation images directory.\n")
    save_dir = input("Please select your save directory.\n")

    # Ensure all directory paths end with a slash
    for string in [image_dir, mask_dir, save_dir]:
        if not string.endswith('/'):
            string += '/'

    # Process each image in the image directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # Construct full paths for image, mask, and output
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('.jpg', '_segmentation.png'))
            output_path = os.path.join(save_dir, filename)

            # Check if corresponding mask exists
            if os.path.exists(mask_path):
                # Open the original image
                original_image = Image.open(image_path)
                
                # Open and process the mask image
                mask_image = Image.open(mask_path)
                mask_rgb = mask_image.convert('RGB')  # Convert mask to RGB
                mask_bw = mask_image.convert('L')     # Convert mask to black and white

                # Create composite image: original image masked by the segmentation
                processed_image = Image.composite(original_image, mask_rgb, mask_bw)

                # Save the processed image
                processed_image.save(output_path)
                print(f"Processed {filename}")
            else:
                print(f"No mask found for {filename}")

# Call the function if this script is run directly
if __name__ == "__main__":
    create_segmented_images()