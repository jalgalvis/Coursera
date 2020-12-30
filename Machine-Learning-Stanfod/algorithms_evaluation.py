import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def add_ones(arr):
    m = arr.shape[0]
    ones = np.ones((m, 1))
    return np.hstack((ones, arr))


def linear_regression_reg_cost(theta, X_ones, y, Lambda):
    m = len(y)
    prediction = X_ones @ theta
    cost = 1 / (2 * m) * np.sum((prediction - y) ** 2)
    theta_reg = np.array(theta)
    theta_reg[0] = 0
    regularization = Lambda / (2 * m) * np.sum(theta_reg ** 2)
    J = cost + regularization
    grad = (1 / m) * (X_ones.T @ (prediction - y))
    regularization = Lambda / m * theta_reg
    grad = (grad + regularization).ravel()
    return J, grad


def train_linear_reg(X_ones, y, Lambda):
    initial_theta = np.zeros(X_ones.shape[1])
    result = opt.minimize(linear_regression_reg_cost, x0=initial_theta, args=(X_ones, y, Lambda), method="BFGS", jac=True)
    theta = result["x"]
    return theta


def learning_curve(X_ones, y, Xval_ones, yval, Lambda):
    m = X_ones.shape[0]
    error_training = np.zeros(m)
    error_validation = np.zeros(m)
    for i in range(1, m+1):
        X_train = X_ones[:i]
        y_train = y[:i]
        theta = train_linear_reg(X_train, y_train, Lambda)
        error_training[i-1], _ = linear_regression_reg_cost(theta, X_train, y_train, Lambda)
        error_validation[i-1], _ = linear_regression_reg_cost(theta, Xval_ones, yval, Lambda)
    return error_training, error_validation


def poly_features(base, degree):
    poly = np.array(base)
    for i in range(2, degree + 1):
        poly = np.hstack((poly, base ** i))
    return poly


def feature_normalization(base):
    mu = base.mean(axis=0)
    sigma = base.std(axis=0)
    base_norm = (base - mu) / sigma
    return base_norm, mu, sigma


def poly_regression_fit_plot(X_ones, X, y, Lambda, mu, sigma, degree):
    theta = train_linear_reg(X_ones, y, Lambda)
    plt.title(f'Polynomial Regression Fit (lambda = {Lambda})')
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    x_values = np.arange(min(X), max(X), 0.05).reshape((-1,1))
    x_values_poly = poly_features(x_values, degree)
    x_values_poly = (x_values_poly - mu)/sigma
    x_values_poly = add_ones(x_values_poly)
    plt.plot(x_values, x_values_poly @ theta, '--', linewidth=2);


def poly_regression_learning_curve_plot(X_ones, y, X_val_ones, yval, Lambda, m):
    error_training, error_validation = learning_curve(X_ones, y, X_val_ones, yval, Lambda)
    p1, p2 = plt.plot(range(m), error_training, range(m), error_validation)
    plt.title(f'Polynomial Regression Learning Curve (lambda = {Lambda})')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend((p1, p2), ('Train', 'Cross Validation'), numpoints=1, handlelength=0.5);


def curve_validation(X_ones, y, X_val_ones, yval, lambda_vec):
    n = len(lambda_vec)
    error_training = np.zeros(n)
    error_validation = np.zeros(n)
    for i in range(n):
        Lambda = lambda_vec[i]
        theta = train_linear_reg(X_ones, y, Lambda)
        error_training[i], _ = linear_regression_reg_cost(theta, X_ones, y, Lambda)
        error_validation[i], _ = linear_regression_reg_cost(theta, X_val_ones, yval, Lambda)
    return error_training, error_validation
