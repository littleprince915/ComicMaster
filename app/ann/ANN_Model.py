import ANN_Functions as annf
import numpy as np


def L_layer_model(X, Y, num_iterations=2000, learning_rate=0.005, print_cost=False, layers_dims=[30000, 20, 7, 5, 1]):
    np.random.seed(1)
    costs = []

    # Parameters initialization.
    parameters = annf.initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in xrange(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = annf.L_model_forward(X, parameters)

        # Compute cost.
        cost = annf.compute_cost(AL, Y)

        # Backward propagation.
        grads = annf.L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = annf.update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters
