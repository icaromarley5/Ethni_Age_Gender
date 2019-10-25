'''
@author: icaromarley5

Script com classes e funções para lidar com o contexto de cada problema: idade, gênero e etnia


https://susanqq.github.io/UTKFace/
Labels
[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
'''

import numpy as np 
from PIL import Image
import os 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

data_path = 'data/'

original_data_path = data_path + 'original/'
train_data_path = data_path + 'train/'
valid_data_path = data_path + 'valid/'
test_data_path = data_path + 'test/'

class EthniProblem():
    name = 'ethnicity'
    def __init__(self): 
        model_path = 'model_{}/'.format(self.name)
        self.input_shape = (75,75,1)
        self.output_shape = 5
        self.train_data = self.load_data('train')
        self.valid_data = self.load_data('valid')
        self.epochs = 500
        self.callbacks = [
            ModelCheckpoint(
                model_path+'{acc:.3f}-{val_acc:.3f}-{epoch:02d}.h5',
                monitor='val_acc',
                verbose=1,
                save_best_only=True),
            EarlyStopping('val_acc',patience = 100)
        ]

    def load_data(self,data_type):
        if data_type == 'train':
            data_path = train_data_path
        else:
            data_path = valid_data_path
        x = os.listdir(data_path)
        target_dict = {
            '0':[1,0,0,0,0],
            '1':[0,1,0,0,0],
            '2':[0,0,1,0,0],
            '3':[0,0,0,1,0],
            '4':[0,0,0,0,1],      
        }
        y = np.array([target_dict[file.split('_')[2]] for file in x])
        x = [data_path + x for x in x] 
        x = np.array([np.array(Image.open(file))
               for file in x])
        x = np.expand_dims(x, axis=3)
        
        return x,y

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(8, (2, 2), 
                        input_shape=(self.input_shape),
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten()) 
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.output_shape, activation='softmax'))

        model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )
        return model
    
class GenderProblem():
    name = 'gender'
    def __init__(self):        
        model_path = 'model_{}/'.format(self.name)
        self.input_shape = (75,75,1)
        self.output_shape = 1
        self.train_data = self.load_data('train')
        self.valid_data = self.load_data('valid')
        self.epochs = 500
        self.callbacks = [
            ModelCheckpoint(
                model_path+'{acc:.3f}-{val_acc:.3f}-{epoch:02d}.h5',
                monitor='val_acc',
                verbose=1,
                save_best_only=True),
            EarlyStopping('val_acc',patience = 100)
        ]

    def load_data(self,data_type):
        if data_type == 'train':
            data_path = train_data_path
        else:
            data_path = valid_data_path
        x = os.listdir(data_path)
        y = np.array([int(file.split('_')[1]) for file in x])
        x = [data_path + x for x in x] 
        x = np.array([np.array(Image.open(file))
               for file in x])
        x = np.expand_dims(x, axis=3)
        
        return x,y
    
    def get_model(self):
        model = Sequential()
        
        model.add(Conv2D(8, (2, 2), 
                        input_shape=(self.input_shape),
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten()) 
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.output_shape, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return model
       
class AgeProblem():
    name = 'age'
    def __init__(self):        
        model_path = 'model_{}/'.format(self.name)
        self.input_shape = (75,75,1)
        self.output_shape = 1
        self.train_data = self.load_data('train')
        self.valid_data = self.load_data('valid')
        self.epochs = 500
        self.callbacks = [
            ModelCheckpoint(
                model_path+'{loss:.3f}-{val_loss:.3f}-{epoch:02d}.h5',
                monitor='val_loss',
                verbose=1,
                save_best_only=True),
            EarlyStopping('val_loss',patience = 100)
        ]

    def load_data(self,data_type):
        if data_type == 'train':
            data_path = train_data_path
        else:
            data_path = valid_data_path
        x = os.listdir(data_path)
        y = np.array([int(file.split('_')[0]) for file in x])
        x = [data_path + x for x in x] 
        x = np.array([np.array(Image.open(file))
               for file in x])
        x = np.expand_dims(x, axis=3)
        
        return x,y
    
    def get_model(self):
        
        model = Sequential()

        model.add(Conv2D(16, (2, 2), 
                        input_shape=(self.input_shape),
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten()) 
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.output_shape, activation='relu'))

        model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        )
        return model
    
def get_problem(problem_name):
    problem_list = [
        GenderProblem,
        AgeProblem,
        EthniProblem
    ]
    for problem in problem_list:
        if problem.name == problem_name:
            return problem()

def process_img(input_path,output_path):
    img = Image.open(input_path)
    img = img.resize((problem.input_shape[0],problem.input_shape[1]), Image.ANTIALIAS)
    img = img.convert('L')
    img.save(output_path) 