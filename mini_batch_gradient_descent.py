# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

def normalization_data(norm_type, data_set):
    """
    :param norm_type:  'l1', 'l2', 'min_max'
    :param data_set:
    :return: normal_data
    """
    if norm_type == 'l1':
        normalizer = preprocessing.Normalizer(norm = norm_type)
        normal_data = normalizer.fit_transform(data_set)
    if norm_type == 'l2':
        normalizer = preprocessing.Normalizer(norm = norm_type)
        normal_data = normalizer.fit_transform(data_set)
    if norm_type == 'min_max':
        min_max_scaler = preprocessing.MinMaxScaler()
        normalizer = min_max_scaler(feature_range = (0, 1))
        normal_data = normalizer.fit_transform(data_set)
    return normal_data

def standardize(x_train_set):
    scaler = preprocessing.StandardScaler().fit(x_train_set)
    standizedX = scaler.transform(x_train_set)
    return standizedX

def load_data(csv_file_name, col_name_list, normalizationType):
    """
    :param csv_file_name:
    :param col_name_list:
    :param normalizationType:
    :return:  feature, target
    """
    data_frame = pd.read_csv(csv_file_name, names = col_name_list)
    data_array = data_frame.values
    feature = data_array[:, :-1]
    target = data_array[:, -1]
    #feature = normalization_data(norm_type = normalizationType, data_set = feature)
    feature = standardize(x_train_set = feature)
    return feature, target

def spilt_data_set(x_data, y_data, test_rate):
    """
    :param xData:
    :param yData:
    :param testRate:
    :return: X_train, X_test, Y_train, Y_test
    """
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, train_size= test_rate, random_state = 42)
    return X_train, X_test, Y_train, Y_test

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def hypothesis(z, w):
    return sigmoid(np.dot(np.c_[np.ones(z.shape[0]), z], w.T)).reshape(z.shape[0],)

def logisitic_gradient(x, y, w):
    """
    :param x:
    :param y:
    :param w:
    :return: loss, gradient
    """
    y_hat = hypothesis(x, w)
    loss = (1.0 / x.shape[0]) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    gradient = np.dot((y_hat - y).T, np.c_[np.ones(x.shape[0]), x]) / x.shape[0]
    return -loss, gradient

def logistic_regression(x, y, w, max_epoch, lr, batch_size, epsilon, momentum = None, adagrad = False, ):
    #Implement optimizer : Momentum, Adagrad
    """
    :param x:
    :param y:
    :param w:
    :param max_epoch:
    :param lr:
    :param batch_size:
    :param epsilon:
    :param momentum:
    :param adagrad:
    :return: loss_history, w
    """
    loss_change = 1.
    loss_history = [1.]
    momentum_vector = [0.]
    gradient_history = [np.ones(9)]
    i = 0
    while i < max_epoch and loss_change > epsilon: #check iteration & convergence
        for j in range(0, x.shape[0], batch_size):
            x_batch = x[j:batch_size + j, :]
            y_batch = y[j:batch_size + j]
            loss, gradient = logisitic_gradient(x_batch, y_batch, w)
            if momentum is not None:
                g_ = momentum * momentum_vector[-1] + lr * gradient
                momentum_vector.append(g_)
            elif adagrad is True:
                g_ = lr * 1. * gradient / np.sqrt(np.sum(gradient_history, axis = 0))
                gradient_history.append(g_)
            else:
                g_ = lr * gradient
            w = w - g_
            loss_history.append(loss)

            if(loss < epsilon):
                break
            i += 1
    return np.array(loss_history), w

def validating_and_result(test_set, weight, str_solver_name):
    # Validation
    value, prob = predict(test_set, weight)
    accuracy = np.sum(value == np.ravel(y_test)) * 1.0 / test_set.shape[0]

    # Print Result: Coefficents and intercept
    print("Training by {}:".format(str_solver_name))
    print("Coefficients: {}".format(weight[0, 1:]))
    print("Intercept: {}".format(weight[0, 0]))
    print("Accuracy: {}".format(accuracy))
    print("")

def predict(x, w):
    predict_prob = sigmoid(np.dot(w, np.c_[np.ones(x.shape[0]), x].T))
    predict_value = np.where(predict_prob > 0.5, 1, 0)
    return predict_value, predict_prob

def sklearn_logistic_regression(x_train, y_train):
    regressor = linear_model.LogisticRegression()
    regressor.fit(x_train, np.ravel(y_train))
    return regressor

if __name__ == '__main__':
    # Loading data from csv file and split to training and testing data set
    data_file = './pima-indians-diabetes.data.csv'
    col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    x_data, y_data = load_data(data_file, col_names, 'l2')
    x_train, x_test, y_train, y_test = spilt_data_set(x_data, y_data, 0.7)
    #print(x_train[0: 100])

    # Initialization training parameters
    max_epoch = 1000 #Maximum iteration to avoid dead loop
    update_epsilon = 1e-5
    learning_rate = 0.01
    train_loss = []
    batch_size = 30

    # Initialize weight
    np.random.seed(0)
    w = np.random.rand(1, x_train.shape[1] + 1)

    #This is a binary-classification problem. Use Logistic Regression
    #Training by using normal mini_batch SGD
    train_loss_mini, trained_w_mini = logistic_regression(x_train, y_train, w, max_epoch, learning_rate, batch_size, update_epsilon)
    validating_and_result(x_test, trained_w_mini, 'Mini Batch SGD')
    # Training by using normal mini_batch SGD with momentum
    train_loss_momentum, trained_w_momentum = logistic_regression(x_train, y_train, w, max_epoch, learning_rate, batch_size,
                                                                  update_epsilon, momentum = 0.9)
    validating_and_result(x_test, trained_w_momentum, 'Mini Batch SGD with Momentum')
    # Training by using normal mini_batch SGD with adagrad
    train_loss_adagrad, trained_w_adagrade = logistic_regression(x_train, y_train, w, max_epoch, learning_rate,
                                                                 batch_size, update_epsilon, adagrad = True)
    validating_and_result(x_test, trained_w_adagrade, 'Mini Batch SGD with Adagrad')

    regressor = sklearn_logistic_regression(x_train, y_train)
    C = regressor.coef_
    # Draw Cost Function loss

    plt.figure(1)
    plt.plot(range(len(train_loss_mini[0:])), train_loss_mini[0:], color='red', label='Mini Batch SGD')
    plt.plot(range(len(train_loss_momentum[0:])), train_loss_momentum[0:], color='blue', label='Mini Batch SGD with Momentum')
    plt.plot(range(len(train_loss_adagrad[0:])), train_loss_adagrad[0:], color='green', label='Mini Batch SGD with Adagrad')

    print('Sklearn Accuracy:{}'.format(np.sum((regressor.predict(x_test) == np.ravel(y_test))) * 1.0 / x_test.shape[0]))
    plt.legend(bbox_to_anchor=(1., 1.), loc=0, borderaxespad=0.)
    plt.show()
