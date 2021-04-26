import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NN import mlp, model_accuracy

df = pd.read_csv("zoo.csv")

# inputs and outputs
inputs = df.drop(columns=['animal_name', 'class_type'])
outputs = pd.get_dummies(df['class_type'].values)
outputs = outputs.values

# 80-20 split for training and testing sets
split = round(.7 * len(inputs))

X_train = np.array(inputs[:split])
X_test = np.array(inputs[split:])

y_train = np.array(outputs[:split])
y_test = np.array(outputs[split:])


model = mlp(inputs=X_train, outputs=y_train, hidden_layer_width=32, learning_rate=0.01, epochs=100000)

hw = model['hidden layer weights']
ow = model['output layer weights']
hb = model['hidden layer biases']
ob = model['output layer biases']

model_accuracy(test_inputs=X_test, test_outputs=y_test, hidden_weights=hw, output_weights=ow,
               hidden_biases=hb, output_biases=ob)
