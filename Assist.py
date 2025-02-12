import os
import csv

def get_filenames(folder_path):
    # Create list to store filenames
    filenames = []
    
    # Iterate through all files in the folder
    for file in os.listdir(folder_path):
        # Split filename and extension, take only filename
        name = os.path.splitext(file)[0]
        filenames.append([name])
    
    # Write to CSV
    with open('filenames.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename'])  # Header
        writer.writerows(filenames)

# Usage
folder_path = 'Dataset/Masks'  # Current directory, change this to your folder path
get_filenames(folder_path)