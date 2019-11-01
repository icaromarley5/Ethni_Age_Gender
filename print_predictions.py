'''
https://docs.microsoft.com/pt-br/visualstudio/python/interactive-repl-ipython?view=vs-2019
https://stackoverflow.com/questions/52310689/use-ipython-repl-in-vs-code
'''

from keras.models import load_model

model_path = 'model_age/'
model_name = '0.86-0.85-27.h5'

model = load_model(model_path+model_name)

predict_dir = 'data/'
load all models

para cada modelo
    print(model.summary())
    printar pontuação de train, valid e teste

para cada imagem no predict dir
    printar previsões dos 3 modelos
    printar valores reais
    printar imagem
    
prints de mocs
