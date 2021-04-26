# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

try:
    import data_utils
    import solver
    import cnn
except Exception:
    import data_utils
    import solver
    import cnn

import numpy as np

# get data
data = data_utils.get_CIFAR10_data()
# initialize model
model = cnn.ThreeLayerConvNet(reg=0.9)
solver = solver.Solver(model, data,
                       lr_decay=0.95,
                       print_every=10, num_epochs=5, batch_size=2,
                       update_rule='sgd_momentum',
                       optim_config={'learning_rate': 5e-4, 'momentum': 0.9})
# train to get the best model
solver.train()

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

best_model = model
y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())
