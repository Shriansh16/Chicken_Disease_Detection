import os
import sys
import yaml
from box import ConfigBox
sys.path.insert(0, 'D:\Chicken_Disease_Classification\src')
from logger import logging
from exception import CustomException
import json
from pathlib import Path
import base64
import joblib
##The primary use case for joblib is to provide a replacement for the pickle module when dealing with large NumPy arrays.
##It is particularly useful for efficiently saving and loading objects containing large numerical arrays

def read_yaml(path_to_yaml):
    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e,sys)

def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    return ConfigBox(content)
'''
if we create dictionary using configbox, it will help us to deal it nested dictionary and accessing will become easier
dict=ConfigBox({'d1':23,'d2':33})
to get the value of d1 simply write
dict.d1    we will get the same output as dict['d1']
'''

def save_bin(data, path: Path):   
    ## to save any file in binary format
    joblib.dump(value=data, filename=path)

def load_bin(path: Path):
    data = joblib.load(path)
    return data

def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
def decodeImage(imgstring, fileName):
    '''
    this function takes a base64-encoded image string (imgstring) and a file name (fileName) as input.
      The function decodes the base64 string and writes the resulting image data to a binary file with the specified file name.
    '''
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    '''
     reads the content of an image file specified by the croppedImagePath parameter,
       encodes it into base64 using base64.b64encode, and then returns the base64-encoded string.
       Base64 is a binary-to-text encoding scheme that represents binary data in an ASCII string format
    '''
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    