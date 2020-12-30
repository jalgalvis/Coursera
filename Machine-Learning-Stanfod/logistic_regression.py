import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    """
    Function that calculates sigmoid factor of any scalar or array
    :param z:   base value for sigmoid
    :return:    sigmoid factor
    """
    return 1 / (1 + np.exp(-z))


def prediction_accuracy(X, y, theta):
    """
    Function that returns the accuracy of a training set and a parameter vector θ
    :param X:   data of parameters X
    :param y:   results for each seat of parameters X
    :param theta:   calculated θ
    :return:    percentage of accuracy
    """
    probability = sigmoid(X @ theta)
    p = probability >= 0.5
    return sum(p == y)


def multi_class_prediction_accuracy(X, y, theta):
    probability = sigmoid(X @ theta)
    prediction = np.argmax(probability, axis=1)
    return np.mean(prediction == y)


def cost_function(theta, X, y):
    """
    Function that calculates the cost and gradient for logistic regression model
    :param theta:   Initial θ
    :param X:       Set of values for parameters X
    :param y:       Set of results for each set of values X
    :return:        J(θ) - cost
    """
    m = len(y)
    prediction = sigmoid(X @ theta)
    J = (1 / m) * ((-y.T @ np.log(prediction)) - (1 - y).T @ np.log(1 - prediction))
    grad = (1 / m) * (X.T @ (prediction - y))
    return J, grad.ravel()


def cost_function_gradient_reg(theta, X, y, Lambda=0):
    """
    Function that calculates the cost and gradient for logistic regression model
    :param theta:   Initial θ
    :param X:       Set of values for parameters X
    :param y:       Set of results for each set of values X
    :return:        cost and gradient
    """
    m = len(y)
    prediction = sigmoid(X @ theta)
    cost = (1 / m) * ((-y.T @ np.log(prediction)) - (1 - y).T @ np.log(1 - prediction))
    regularization = (Lambda / (2 * m)) * (theta ** 2)
    J = cost + regularization
    theta[0] = 0
    gradient = (1 / m) * (X.T @ (prediction - y)) + (Lambda / m) * theta

    return J[0], gradient.ravel()


def gradient_descent(X, y, theta, alpha, num_iters, Lambda):
    """
    Function that calculates the gradient descent or optimized θ for logistic regression model
    :param X:       Set of values for parameters X
    :param y:       Set of results for each set of values X
    :param theta:   Initial θ
    :param alpha:   Value of parameter ⍺
    :param num_iters:   Number of iterations
    :param Lambda: Value of paramenter λ
    :return:        cost and gradient
    """
    J_history = []

    for i in range(num_iters):
        cost, grad = cost_function_gradient_reg(theta, X, y, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history


def multi_class_theta_optimization(X, y, Lambda, gradient_decent=False, iterations=0, alpha=1):
    """
    Function that calculates θ optimiation for multi-class classification with logistic regression model
    :param X:       Set of values for parameters X
    :param y:       Set of results for each set of values X
    :param Lambda: Value of paramenter λ
    :param gradient_decent: If true it uses gradient descent for global optima, otherwise uses fmin_tnc from scipy
    :param iterations:  Number of iterations for gradient decent
    :param alpha:   Value of parameter ⍺ for gradient decent
    :return:        optimized theta
    """
    labels = np.unique(y)
    initial_theta = np.zeros((X.shape[1], 1))
    theta_opt = np.empty((0, X.shape[1]))
    for i in labels:
        sel_y = (y == i).astype(int).reshape(-1, 1)
        if gradient_decent:
            result = gradient_descent(X, sel_y, initial_theta, alpha=alpha, num_iters=iterations, Lambda=Lambda)
        else:
            result = opt.fmin_tnc(func=cost_function_gradient_reg, x0=initial_theta, args=(X, sel_y, Lambda))
        new_theta = result[0]
        theta_opt = np.vstack((theta_opt, new_theta.reshape(1, -1)))
    return theta_opt.T


def normal_equation_reg(X, y, Lambda):
    """
    Function that calculates the normal equation or optimized θ for logistic regression model
    :param X:       Set of values for parameters X
    :param y:       Set of results for each set of values X
    :param Lambda: Value of paramenter λ
    :return:        optimum theta
    """
    L = np.eye(X.shape[1])
    L[0][0] = 0
    L = L * Lambda
    f1 = np.linalg.pinv(X.transpose() @ X + L)
    f2 = X.transpose() @ y
    theta = f1 @ f2

    return theta


def plot_scatter(X, y, label_y1, label_y2):
    """
    Function to draw a customed scatter plot
    :param X:   Values X
    :param y:   Values y
    :param label_y1:    Label values X
    :param label_y2:    Label values y
    """
    i = np.where(y == 1)
    j = np.where(y == 0)
    plt.scatter(x=X[i, 0], y=X[i, 1], marker='*', label=label_y1)
    plt.scatter(x=X[j, 0], y=X[j, 1], marker='o', label=label_y2)


def map_feature(x1, x2, degree):
    """
    map the features into all polynomial terms of the x1 and x2 parameters up to the power defined in the degree argument.
    :param x1:  First parameter
    :param x2:  Second parameter parameter
    :param degree: degree of power
    :return: higher=dimensional feature vector
    """
    out = np.ones(1) if x1.shape == () else np.ones(len(x1)).reshape(-1, 1)

    for i in range(1, degree + 1):
        for j in range(i + 1):
            new_term = x1 ** (i - j) * x2 ** j
            if x1.shape != ():
                new_term = new_term.reshape(-1, 1)
            out = np.hstack((out, new_term))
    return out
