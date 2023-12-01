import os

def delete_png_files(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    file_count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            file_count += 1

    print(f"Deleted {file_count} PNG files from {directory}")

directory = 'C:/Users/Alienware/Desktop/github/projectdata/output/'

delete_png_files(directory)
