import os
import shutil

def merge_folders(source_parent_folder, destination_parent_folder):
    subfolders = [folder for folder in os.listdir(source_parent_folder) if os.path.isdir(os.path.join(source_parent_folder, folder))]
    folder_dict = {}

    # Group subfolders by their base name (excluding the number)
    for subfolder in subfolders:
        base_name = ''.join([i for i in subfolder if not i.isdigit()])
        if base_name not in folder_dict:
            folder_dict[base_name] = []
        folder_dict[base_name].append(subfolder)

    for base_name, folders in folder_dict.items():
        result_folder_name = os.path.join(destination_parent_folder, f"{base_name}")

        # Create a new subfolder for the merged folders
        os.makedirs(result_folder_name, exist_ok=True)

        # Iterate through folders with the same base name
        for folder in folders:
            folder_path = os.path.join(source_parent_folder, folder)

            # Iterate through files in the subfolder
            for file_name in os.listdir(folder_path):
                source_path = os.path.join(folder_path, file_name)
                destination_path = os.path.join(result_folder_name, file_name)

                # If the file already exists in the result folder, append a number to the filename
                counter = 1
                while os.path.exists(destination_path):
                    file_name, file_extension = os.path.splitext(file_name)
                    new_file_name = f"{file_name}_{counter}{file_extension}"
                    destination_path = os.path.join(result_folder_name, new_file_name)
                    counter += 1

                # Copy the file to the result folder
                shutil.copy2(source_path, destination_path)

        print(f"Subfolders with base name '{base_name}' merged into '{result_folder_name}'.")

if __name__ == "__main__":
    source_folder = 'C:/Users/brutchjw/Documents/CSSE463/Project/LogoDet-3K (1)/LogoDet-3K/Transportation'
    destination_folder = 'C:/Users/brutchjw/Documents/CSSE463/Project/TransportationCleaned'

    # Call the function to merge subfolders and copy to the destination folder
    merge_folders(source_folder, destination_folder)