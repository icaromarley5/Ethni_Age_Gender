# -*- coding: utf-8 -*-
"""
@author: icaromarley5

Script para treinamento das redes
"""
from problem import get_problem

# gender
# age
# ethnicity
problem_name = 'age'

problem = get_problem(problem_name)
model = problem.get_model()
X_train,y_train = problem.train_data

model.fit(
        x = X_train,y=y_train,
        epochs=problem.epochs,
        validation_data=problem.valid_data ,
        callbacks=problem.callbacks
)