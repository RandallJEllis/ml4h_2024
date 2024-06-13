import os
import pickle

def check_folder_existence(directory_path):
    # Check if the directory exists and create it if it does not
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_pickle(filepath, data):
    # Check if the directory exists
    last_slash = filepath.rfind('/')
    folder = filepath[:last_slash]

    check_folder_existence(folder)
    
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)