from PIL import Image
import os

def check_image_dimensions(directory_path):
    """
    Check all images in a directory to identify which ones are 1024x1024 and which are not.
    
    Args:
        directory_path (str): Path to the directory containing images
        
    Returns:
        tuple: Two lists containing paths of matching and non-matching images
    """
    matching_images = []
    non_matching_images = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(directory_path, filename)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width == 1024 and height == 1024:
                        matching_images.append(image_path)
                    else:
                        non_matching_images.append(image_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return matching_images, non_matching_images

if __name__ == "__main__":
    # Replace this with your directory path
    directory = "Dataset/Masks"
    
    matching, non_matching = check_image_dimensions(directory)
    
    # Print matching images with their dimensions
    print("\n=== Images that are 1024x1024 ===")
    for img_path in matching:
        with Image.open(img_path) as img:
            print(f"✓ {os.path.basename(img_path)}")
    
    # Print non-matching images with their dimensions
    print("\n=== Images that are NOT 1024x1024 ===")
    for img_path in non_matching:
        with Image.open(img_path) as img:
            width, height = img.size
            print(f"✗ {os.path.basename(img_path)} - {width}x{height}")
    
    # Print summary
    total_images = len(matching) + len(non_matching)
    print(f"\n=== Summary ===")
    print(f"Total images checked: {total_images}")
    print(f"Images matching 1024x1024: {len(matching)}")
    print(f"Images not matching: {len(non_matching)}")