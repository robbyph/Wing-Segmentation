import os
import re

def rename_photos(directory):
    # Regular expression pattern to match 'M' followed by exactly 3 digits
    pattern = r'M\d{3}'
    
    # List all files in the directory
    for filename in os.listdir(directory):
        # Get the full path of the file
        file_path = os.path.join(directory, filename)
        
        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue
            
        # Search for the pattern in the filename
        match = re.search(pattern, filename)
        if match:
            # Get the matched pattern (M followed by 3 digits)
            new_name = match.group(0)
            
            # Get the file extension
            _, ext = os.path.splitext(filename)
            
            # Create the new filename with the original extension
            new_filename = new_name + ext
            new_file_path = os.path.join(directory, new_filename)
            
            # Rename the file if the new filename doesn't already exist
            if not os.path.exists(new_file_path):
                try:
                    os.rename(file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")
                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")
            else:
                print(f"Skipped '{filename}': '{new_filename}' already exists")

# Example usage
if __name__ == "__main__":
    # Replace with your directory path
    photo_directory = "Dataset/Masks"
    rename_photos(photo_directory)