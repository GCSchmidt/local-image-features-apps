import os
import sys
import subprocess
import platform
import zipfile
import tarfile

def create_virtual_environment():
    """Creates a virtual environment named '.venv'."""
    if not os.path.exists(".venv"):
        try:
            # Use sys.executable for cross-platform compatibility
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
            print("Virtual environment '.venv' created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False
    else:
        print("Virtual environment '.venv' already exists. Skipping creation.")
    return True


def install_requirements():
    """installs requirements into virtual environment."""
    try:

        if platform.system() == "Windows":
            python_path = os.path.join(".venv", "Scripts", "python.exe")
        else:
            python_path = os.path.join(".venv", "bin", "python")

        if not os.path.exists(python_path):
            raise FileNotFoundError(f"Virtual environment Python executable not found: {python_path}")     

        requirements_file = "requirements.txt"
        if os.path.exists(requirements_file):
            subprocess.run([python_path, "-m", "pip", "install", "-r", requirements_file], check=True)
            print("Required libraries installed successfully in the virtual environment.")
        else:
            print(f"Requirements file '{requirements_file}' not found. Skipping installation.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing required libraries: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def unzip_dataset(origin: str, destionation: str):
    """Unzips a file at path to destionation"""
    try:
        if origin.endswith('.tgz'):
            with tarfile.open(origin) as file:
                file.extractall(path=destionation)
        print("Dataset unzipped successfully.")
    except FileNotFoundError:
        print("sudoku.zip not found in datasets folder.")
    except Exception as e:
        print(f"Error unzipping dataset: {e}")


if __name__ == "__main__":
    if create_virtual_environment():
        install_requirements()

    unzip_dataset(r'./image-stitching/datasets/example-data.tgz', r'./image-stitching/datasets')
