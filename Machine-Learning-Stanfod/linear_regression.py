import numpy as np
import matplotlib.pyplot as plt

params = {"ytick.color": "w",
          "xtick.color": "w",
          "axes.labelcolor": "w",
          "axes.edgecolor": "y",
          "axes.facecolor": "None",
          "text.color": 'w'}
plt.rcParams.update(params)


def compute_cost(X, y, theta):
    prediction = np.matmul(X, theta)
    m = len(X)
    J = 1 / (2 * m) * sum((prediction - y) ** 2)
    return J


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(X)
    J_history = np.zeros((iterations, 1))
    for iteration in range(iterations):
        prediction = X @ theta
        j = 1 / m * sum((prediction - y) * X)
        theta = theta - alpha * j.reshape(-1, 1)
        J_history[iteration] = compute_cost(X=X, y=y, theta=theta)
    return theta, J_history


def feature_normalization(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def plot_learning_curve(X, y, num_iters):
    theta = np.zeros((3, 1))
    for alpha in (0.3, 0.1, 0.03, 0.01):
        _, J_hist = gradient_descent(X, y, theta, alpha, num_iters)
        plt.plot(np.arange(num_iters), J_hist, label=f'alpha: {alpha}');
    plt.xlabel('Number if iternations')
    plt.ylabel('Cost J')
    plt.title(f'iterations: {num_iters}')
    plt.legend()
