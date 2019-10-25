# -*- coding: utf-8 -*-
"""
@author: icaromarley5

https://susanqq.github.io/UTKFace/
Labels
The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
"""

import numpy as np 
from PIL import Image
import os 

data_path = 'data/'

original_data_path = data_path + 'original/'
train_data_path = data_path + 'train/'
valid_data_path = data_path + 'valid/'
test_data_path = data_path + 'test/'

class DataLoader():
    def __init__(self, data_type):
        if data_type == 'train':
            self.data_path = train_data_path
        else:
            self.data_path = valid_data_path
            
class AgeDataLoader(DataLoader):
    input_shape = (75,75,1)
    output_shape = 1
    def __init__(self, data_type):
        super().__init__(data_type)
        self.x = os.listdir(self.data_path)
        self.y = np.array([int(file.split('_')[0]) for file in self.x])
        self.x = [self.data_path + x for x in self.x] 
        self.x = np.array([np.array(Image.open(file))
               for file in self.x])
        self.x = np.expand_dims(self.x, axis=3)

class GenderDataLoader(DataLoader):
    input_shape = (75,75,1)
    output_shape = 1
    def __init__(self, data_type):
        super().__init__(data_type)
        self.x = os.listdir(self.data_path)
        self.y = np.array([int(file.split('_')[1]) for file in self.x])
        self.x = [self.data_path + x for x in self.x] 
        self.x = np.array([np.array(Image.open(file))
               for file in self.x])
        self.x = np.expand_dims(self.x, axis=3)
          
class EthniDataLoader(DataLoader):
    input_shape = (75,75,1)
    output_shape = 5
    def __init__(self, data_type): 
        super().__init__(data_type)
        self.x = os.listdir(self.data_path)
        target_dict = {
            '0':[1,0,0,0,0],
            '1':[0,1,0,0,0],
            '2':[0,0,1,0,0],
            '3':[0,0,0,1,0],
            '4':[0,0,0,0,1],      
        } 
        self.y = np.array([target_dict[file.split('_')[2]] for file in self.x])
        self.x = [self.data_path + x for x in self.x] 
        self.x = np.array([np.array(Image.open(file))
               for file in self.x])
        self.x = np.expand_dims(self.x, axis=3)
    
def process_img(sequence,input_path,output_path):
    img = Image.open(input_path)
    img = img.resize((sequence.input_shape[0],sequence.input_shape[1]), Image.ANTIALIAS)
    img = img.convert('L')
    img.save(output_path) 
    
def get_loader(data_name):
    return {
        'gender':GenderDataLoader(),
        'age':AgeDataLoader(),
        'ethnicity':EthniDataLoader(),
    }.get(data_name)