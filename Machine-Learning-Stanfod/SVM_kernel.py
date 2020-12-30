import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_scatter(X, y, label_y1='y==1', label_y2='y==0', xlabel='xlabel', ylabel='ylabel', title=None):
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

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def svm_train(X, y, C, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    """Trains an SVM classifier"""

    y = y.flatten()  # prevents warning

    # alternative to emulate mapping of 0 -> -1 in svmTrain.m
    #  but results are identical without it
    # also need to cast from unsigned int to regular int
    # otherwise, contour() in visualizeBoundary.py doesn't work as expected
    # y = y.astype("int32")
    # y[y==0] = -1

    if kernelFunction == "gaussian_rbf":
        clf = svm.SVC(C=C, kernel="precomputed", tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(gaussian_kernel_gram_matrix(X, X, sigma=sigma), y)

    else:
        clf = svm.SVC(C=C, kernel=kernelFunction, tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(X, y)


def gaussian_kernel(x1, x2, sigma=0.1):
    # RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()

    sim = np.exp(- np.sum((x1 - x2) ** 2) / float(2 * (sigma ** 2)))

    return sim


def gaussian_kernel_gram_matrix(X1, X2, K_function=gaussian_kernel, sigma=0.1):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2, sigma)
    return gram_matrix


def visualize_boundary_linear(X, y, model):
    """Plots a linear decision boundary learned by the SVM"""

    plt.figure()
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yp = - (w[0] * xp + b) / w[1]

    plt.plot(xp, yp, 'b-', color='tomato')


def visualize_boundary(X, y, model, varargin=0):
    """Plots a non-linear decision boundary learned by the SVM"""

    plt.figure()

    x1plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).T
    x2plot = np.linspace(X[:, 1].min(), X[:, 1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gaussian_kernel_gram_matrix(this_X, X))

    plt.contour(X1, X2, vals, colors="white", levels=[0, 0], linewidth=5)


def dataset_3_params(X, y, Xval, yval):
    """returns optimal C and sigma found in the cross validation set"""

    predictionErrors = np.zeros((64, 3))
    predictionsCounter = 0

    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            print(f" trying C={C:.2f}, sigma={sigma:.2f}...")

            model = svm_train(X, y, C, "gaussian_rbf", sigma=sigma)

            predictions = model.predict(gaussian_kernel_gram_matrix(Xval, X))

            predictionErrors[predictionsCounter, 0] = np.mean((predictions != yval).astype(int))

            predictionErrors[predictionsCounter, 1] = C
            predictionErrors[predictionsCounter, 2] = sigma

            predictionsCounter = predictionsCounter + 1

    print()

    # sort ndarray by first column, from https://stackoverflow.com/a/2828121/583834
    predictionErrors = predictionErrors[predictionErrors[:, 0].argsort()]

    print('\t# Pred. error\tsigma\tC\n')
    for i in range(predictionErrors.shape[0]):
        pred_error, sigma, C, = predictionErrors[i][:]
        print(f'  \t{pred_error:f}\t{sigma:.2f}\t{C:.2f}')

    C = predictionErrors[0, 1]
    sigma = predictionErrors[0, 2]

    return C, sigma
