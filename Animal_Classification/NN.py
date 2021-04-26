import numpy as np
import matplotlib.pyplot as plt


def cross_entropy_back_prop(inputs, outputs, hw, ow, hb, ob, learning_rate, ah, y):
    # softmax gradient
    softmax_der = (y - outputs) / outputs.shape[0]

    # output layer weights and biases gradients
    ow_der = np.dot(ah.T, softmax_der)
    ob_der = np.sum(softmax_der, axis=0, keepdims=True)

    # hidden layer weights and biases gradients
    ah_der = np.dot(softmax_der, ow.T)
    ah_der[ah <= 0] = 0
    hw_der = np.dot(inputs.T, ah_der)
    hb_der = np.sum(ah_der, axis=0, keepdims=True)

    # update output layer and hidden layer weights and biases
    ow -= learning_rate * ow_der
    hw -= learning_rate * hw_der

    ob -= learning_rate * ob_der
    hb -= learning_rate * hb_der


def mlp(inputs, outputs, hidden_layer_width, learning_rate, epochs):
    d = dict()

    # hyperparameters
    np.random.seed(42)

    # weights for hidden and output layers
    hw = np.random.rand(len(inputs[0]), hidden_layer_width)

    ow = np.random.rand(hidden_layer_width, 7)

    # initialize biases for hidden and output layers to zeros
    hb = np.zeros((1, hidden_layer_width))
    ob = np.zeros((1, 7))

    for epoch in range(1, epochs + 1):
        # feed forward
        # hidden layer
        zh = np.dot(inputs, hw) + hb
        ah = relu(zh)

        # output layer
        zo = np.dot(ah, ow) + ob
        y = softmax(zo)

        # calculate cross entropy
        indices = np.argmax(outputs, axis=1).astype(int)
        predicted_probability = y[np.arange(len(y)), indices]
        log_preds = np.log(predicted_probability)
        cross_entropy = -np.sum(log_preds) / len(log_preds)

        # backpropagation
        cross_entropy_back_prop(inputs, outputs, hw, ow, hb, ob, learning_rate, ah, y)

        if epoch == epochs:
            print("After %d epochs, cross entropy is now: %f" % (epoch, cross_entropy))
            d['hidden layer weights'] = hw
            d['output layer weights'] = ow
            d['hidden layer biases'] = hb
            d['output layer biases'] = ob

    return d


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x)
    softmax = e_x / e_x.sum(axis=1, keepdims=True)
    return softmax


def model_accuracy(test_inputs, test_outputs, hidden_weights, output_weights, hidden_biases, output_biases):
    # hidden layer
    zh = np.dot(test_inputs, hidden_weights) + hidden_biases
    ah = relu(zh)

    # output layer
    zo = np.dot(ah, output_weights) + output_biases
    y = softmax(zo)

    preds_correct_boolean = np.argmax(y, 1) == np.argmax(test_outputs, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = correct_predictions / len(test_outputs)
    print("Accuracy is: %f" % accuracy)
