import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "MusicGenreClassification"

list_of_files = [
    f"src/{project_name}/__init__.py",

    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/data_augmentation.py",
    f"src/{project_name}/components/prediction.py",

    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/ann.py",
    f"src/{project_name}/models/cnn.py",
    f"src/{project_name}/models/lstm.py",
    f"src/{project_name}/models/hybrid.py",

    f"src/{project_name}/logger.py",

    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    ".gitignore"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")