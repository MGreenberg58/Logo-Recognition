import os
import shutil

def merge_folders(source, destination):
    # Merge the contents of source folder into destination folder
    for item in os.listdir(source):
        source_path = os.path.join(source, item)
        destination_path = os.path.join(destination, item)

        if os.path.isdir(source_path):
            # If it's a directory, recursively merge its contents
            merge_folders(source_path, destination_path)
        else:
            # If it's a file, copy it to the destination folder
            shutil.copy2(source_path, destination_path)


def process_folders(path, dest):
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            original_name = directory

            # Replace spaces with underscores
            new_name = original_name.replace(" ", "_")

            # Truncate name before "=" if "-" is present
            if "-" in new_name:
                new_name = new_name.split("-")[0]

            # Full path of the original and new folders
            original_path = os.path.join(root, original_name)
            new_path = os.path.join(dest, new_name)

            # Handle naming conflicts by merging folders
            if os.path.exists(new_path):
                merge_folders(original_path, new_path)
            else:
                # If the folder doesn't exist, create a new copy
                shutil.copytree(original_path, new_path)

if __name__ == "__main__":
    source_folder = "E:/Logos/folder/Transportation"
    destination_folder = "E:/Logos/TransportationMerged"

    # Call the function to merge subfolders and copy to the destination folder
    process_folders(source_folder, destination_folder)