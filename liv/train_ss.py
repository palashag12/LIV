import os
import subprocess

def run_training_script(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            command = ["python", "train_liv.py", "training=finetune", f"dataset=LIBERO10/{filename}"]
            print(f"Running: {' '.join(command)}")
            subprocess.run(command)

if __name__ == "__main__":
    folder_path = '/home/pa1077/LIV/liv/cfgs/dataset/LIBERO10'  # Change this to the actual folder path
    run_training_script(folder_path)