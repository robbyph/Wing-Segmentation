import os
from PIL import Image  # We'll use the Pillow library for image processing

def standardize_images(directory_path):
    """
    Convert all .jpeg and .png files to .jpg in the specified directory.
    """
    # List of extensions we want to convert
    convert_extensions = ('.jpeg', '.png')
    
    # Get all files in the directory
    for filename in os.listdir(directory_path):
        # Convert filename to lowercase for checking extensions
        lower_filename = filename.lower()
        # Check if the file has any of our target extensions
        if lower_filename.endswith(convert_extensions):
            try:
                # Construct full file paths
                old_path = os.path.join(directory_path, filename)
                # Remove the old extension and add .jpg
                # We split at the dot and take everything before it
                new_filename = os.path.splitext(filename)[0] + '.jpg'
                new_path = os.path.join(directory_path, new_filename)
                
                # Open and save the image in jpg format
                with Image.open(old_path) as img:
                    # Convert to RGB mode if it's a PNG (in case it has transparency)
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        # Convert to RGB mode to remove transparency
                        img = img.convert('RGB')
                    # Preserve the original image quality
                    img.save(new_path, 'JPEG', quality=95)
                
                # Remove the original file
                os.remove(old_path)
                print(f"Converted: {filename} -> {new_filename}")
            
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

if __name__ == "__main__":
    # Get the directory path from user
    directory = "Dataset/Masks"
    
    standardize_images(directory)
    print("Conversion completed!")