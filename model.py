import numpy as np
import numba

from kernels import kernel_matrix, gram_matrix


@numba.njit()
def fit(xs, ys, kernel_func, epochs, num_classes, dim, include_error=True):
    """
    Fit multi-class kernel perceptron to train data

    :param xs:            train set
    :param ys:            train classes
    :param kernel_func:   kernel function
    :param epochs:        number of epochs to train model
    :param num_classes:   number of classes
    :param dim: dimension used for kernel function
    :param include_error: boolean to include error on train set

    :return: weights, train error rate
    """
    weights = np.zeros((num_classes, len(xs)))
    kernel = gram_matrix(xs, dim, kernel_func)
    for epoch in range(epochs):
        num_correct = 0
        for index, xt in enumerate(xs):
            correct = ys[index]
            vector = np.zeros(num_classes)
            for k in range(num_classes):
                vector[k] = np.dot(weights[k, :], kernel[index, :])
            predicted = np.argmax(vector)
            if correct != predicted:
                weights[correct, index] = weights[correct, index] + 1
                weights[predicted, index] = weights[predicted, index] - 1
            else:
                num_correct += 1
        e = 1 - num_correct / len(xs)
        if e == 0.0:
            break
    error = predict(weights, xs, xs, ys, kernel_func, dim, kernel) if include_error else None
    return weights, error


@numba.njit()
def predict(weights, train_xs, test_xs, test_ys, kernel_func, dim, kernel=None):
    """
    Predict classes of test set using multi-class kernel perceptron algorithm

    :param weights:     weights fit to train set
    :param train_xs:    train set
    :param test_xs:     test set
    :param test_ys:     test classes
    :param kernel_func: kernel function
    :param dim:         dimension used for kernel function
    :param kernel:      kernel matrix

    :return: test error
    """
    K = kernel if kernel is not None else kernel_matrix(test_xs, train_xs, dim, kernel_func)
    num_correct = 0
    w = (weights @ K.T).T
    for index, x in enumerate(test_xs):
        predicted = np.argmax(w[index, :])
        correct = test_ys[index]
        if predicted == correct:
            num_correct += 1
    return 1 - num_correct / len(test_xs)
