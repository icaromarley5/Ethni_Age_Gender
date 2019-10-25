'''
https://docs.microsoft.com/pt-br/visualstudio/python/interactive-repl-ipython?view=vs-2019
https://stackoverflow.com/questions/52310689/use-ipython-repl-in-vs-code
'''
from keras.models import load_model

model_path = 'model/'
model_name = '0.86-0.85-27.h5'

model = load_model(model_path+model_name)

print(model.summary())