import numpy as np
from decimal import Decimal


def unroll_parameters(*args):
    unrolled = np.empty(0)
    for arg in args:
        unrolled = np.hstack((unrolled, arg.reshape(arg.size, order='F')))
    return unrolled


def sigmoid(z):
    """
    Function that calculates sigmoid factor of any scalar or array
    :param z:   base value for sigmoid
    :return:    sigmoid factor
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """
    Function that computes the gradient of the sigmoid function evaluated at z
    :param z:   base value for sigmoid
    :return:    gradient of the sigmoid
    """
    g = 1 / (1 + np.exp(-z))
    return g * (1 - g)


def neural_network_prediction(theta1, theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    probability1 = sigmoid(X @ theta1.T)
    n = probability1.shape[0]
    probability1 = np.hstack((np.ones((n, 1)), probability1))
    probability2 = sigmoid(probability1 @ theta2.T)
    prediction = np.argmax(probability2, axis=1)
    # offsets python's zero notation
    prediction += 1
    prediction[prediction == 10] = 0
    return prediction


def nn_cost_function_reg(nn_params, input_layer_size, hidden_layer_size, X, y, Lambda):
    """
    Implements the neural network cost function for a two layer neural network which performs classification
    Computes the cost and gradient of the neural network. The
    parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices.

    The returned parameter grad should be a "unrolled" vector of the
    partial derivatives of the neural network.
    :param nn_params:   Unrolled initial_thetas
    :param input_layer_size:    size of the input layer
    :param hidden_layer_size:   size of the hidden layer
    :param X:   X
    :param y:   y
    :param Lambda:  Lambda
    :return:    cost and gradient
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    labels = np.unique(y).astype(int)
    theta1 = np.array(np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F'))
    theta2 = np.array(np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (labels.size, hidden_layer_size + 1), order='F'))
    m = X.shape[0]
    # add column of ones as bias unit from input layer to second layer
    a1 = np.hstack((np.ones((m, 1)), X))
    # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    # add column of ones as bias unit from second layer to third layer
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    # offsets python's zero notation (in mathlab indexes start at 1)
    # y[y == 10] = 0
    # for every label, convert it into vector of 0s and a 1 in the appropriate position
    y_matrix = np.empty((len(y), 0))
    for i in labels:
        newy = (y == i).astype(int)
        y_matrix = np.hstack((y_matrix, newy.reshape(-1, 1)))
    # at this point, both a3 and y are m x k matrices, where m is the number of inputs
    # and k is the number of hypotheses. Given that the cost function is a sum
    # over m and k, loop over m and in each loop, sum over k by doing a sum over the row
    J = 1/m * np.sum((-y_matrix * np.log(a3)) - (1 - y_matrix) * np.log(1 - a3))
    # % REGULARIZED COST FUNCTION
    # note that Theta1[:,1:] is necessary given that the first column corresponds to transitions
    # from the bias terms, and we are not regularizing those parameters. Thus, we get rid
    # of the first column.
    sum_square_theta1 = np.sum(theta1[:, 1:] ** 2)
    sum_square_theta2 = np.sum(theta2[:, 1:] ** 2)
    regularization = Lambda / (2 * m) * (sum_square_theta1 + sum_square_theta2)
    J = J + regularization

    # BACKPROPAGATION
    theta1[:, 0] = np.zeros(theta1[:, 0].shape)
    theta2[:, 0] = np.zeros(theta2[:, 0].shape)
    d3 = a3 - y_matrix
    theta2_grad = 1/m * (a2.T @ d3).T + Lambda/m * theta2
    z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
    d2 = (d3 @ theta2) * sigmoid_gradient(z2)
    theta1_grad = 1 / m * (a1.T @ d2).T
    theta1_grad = theta1_grad[1:, :] + Lambda/m * theta1

    grad = unroll_parameters(theta1_grad, theta2_grad)

    return J, grad


def rand_initialize_weights(L_in, L_out):
    """
    Randomly initialize the weights of a layer with L_in incoming connections and L_out outgoing connections

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as the column row of W handles the "bias" terms

    :param L_in:    input layer size
    :param L_out:   output layer size
    :return:
    """

    # Initialize W randomly so that we break the symmetry while training the neural network.
    # The first row of W corresponds to the parameters for the bias units
    #

    # Randomly initialize the weights to small values
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init

    return W


def check_NN_gradients(lambda_reg=0):
    # CHECKNNGRADIENTS Creates a small neural network to check the
    # backpropagation gradients
    #   CHECKNNGRADIENTS(lambda_reg) Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using computeNumericalGradient). These two gradient computations should
    #   result in very similar values.
    #

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(range(m), num_labels).T
    # Unroll parameters
    nn_params = unroll_parameters(theta1, theta2)

    # Short hand for cost function
    def costFunc(p):
        return nn_cost_function_reg(p, input_layer_size, hidden_layer_size, X, y, lambda_reg)

    _, grad = costFunc(nn_params)

    numgrad = compute_numerical_gradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    # code from http://stackoverflow.com/a/27663954/583834
    fmt = '{:<25}{}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

    print('The above two columns you get should be very similar.\n' \
          '(Left Col.: Your Numerical Gradient, Right Col.: Analytical Gradient)')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = Decimal(np.linalg.norm(numgrad - grad)) / Decimal(np.linalg.norm(numgrad + grad))

    print('If your backpropagation implementation is correct, then \n' \
          'the relative difference will be small (less than 1e-9). \n' \
          '\nRelative Difference: {:.10E}'.format(diff))


def debug_initialize_weights(fan_out, fan_in):
    """
    initializes the weights of a layer with fan_in incoming connections and fan_out outgoing connections using a fix set of values

    Note that W should be set to a matrix of size(1 + fan_in, fan_out) as the first row of W handles the "bias" terms

    :param fan_out: Incoming connections
    :param fan_in:  Outcoming connection
    :return:    Initial theta with same values
    """

    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10

    return W


def compute_numerical_gradient(J, theta):
    """
    computes the numerical
    gradient of the function J around theta. Calling y = J(theta) should

    Notes: The following code implements numerical gradient checking, and
           returns the numerical gradient.It sets numgrad(i) to (a numerical
           approximation of) the partial derivative of J with respect to the
           i-th input argument, evaluated at theta. (i.e., numgrad(i) should
           be the (approximately) the partial derivative of J with respect
           to theta(i).)
    :param J:       cost
    :param theta:   theta
    :return:        the function value at theta
    """

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2 * e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad
