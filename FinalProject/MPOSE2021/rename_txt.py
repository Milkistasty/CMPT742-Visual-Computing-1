import os

def rename_files(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    for filename in os.listdir(directory):
        if len(filename) > 8:
            new_filename = filename[8:] 
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

directory = 'C:/Users/Alienware/Desktop/github/projectdata/output/'

rename_files(directory)
