import os
from PIL import Image  # We'll use the Pillow library for image processing

def standardize_images(directory_path):
    """
    Convert all .jpeg files to .jpg in the specified directory.
    """
    # Get all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a .jpeg
        if filename.lower().endswith('.jpeg'):
            try:
                # Construct full file paths
                old_path = os.path.join(directory_path, filename)
                new_path = os.path.join(directory_path, filename[:-5] + '.jpg')
                
                # Open and save the image in jpg format
                with Image.open(old_path) as img:
                    # Preserve the original image quality
                    img.save(new_path, 'JPEG', quality=95)
                
                # Remove the original .jpeg file
                os.remove(old_path)
                print(f"Converted: {filename} -> {filename[:-5]}.jpg")
            
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

if __name__ == "__main__":
    # Get the directory path from user
    directory = "Dataset/Masks"
    
    standardize_images(directory)
    print("Conversion completed!")