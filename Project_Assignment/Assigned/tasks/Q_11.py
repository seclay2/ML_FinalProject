def Q_11(self, X_train_scaled, X_test_scaled, y_train, y_test, learning_rate=0.001, nIteration=10):
    # Task 11: Given the (X_train, y_train) pairs denoting input matrix and output vector respectively,
    # Fit a linear regression model using the batch gradient descent algorithm you learned in class to obtain
    # the coefficients, beta's, as a numpy array of m+1 values (Please recall class lecture).
    # Please use the learning_rate and nIteration (number of iterations) parameters in your implementation
    #  of the gradient descent algorithm.
    # Please measure the cpu_time needed during the training step. cpu_time is not equal to the wall_time. So,
    # use time.perf_counter() for an accurate measurement. Documentation on this function can be found here:
    # https://docs.python.org/3/library/time.html
    # Then using the computed beta values, predict the test samples provided in the "X_test_scaled"
    # argument, and let's call your prediction "y_pred".
    # Compute Root Mean Squared Error (RMSE) of your prediction.
    # Finally, return the beta vector, y_pred, RMSE, cpu_time as a tuple.
    # PLEASE DO NOT USE ANY LIBRARY FUNCTION THAT DOES THE LINEAR REGRESSION.

    import random
    random.seed(554433)
    beta = []
    y_pred = []
    RMSE = -1
    cpu_time = 0

    ## YOUR CODE HERE ###
    import numpy as np
    # np.random.seed(554433)
    from time import perf_counter

    x_temp = np.c_[np.ones((len(np.array(X_train_scaled)), 1)), np.array(X_train_scaled)]  # m+1

    # Using built in Numpy stuff
    # beta = np.random.rand(len(x_temp.columns), 1)  # Don't use this (not a dataframe)
    # beta = np.random.rand(len(x_temp[0]), 1)  # use this because numpy array

    # Using the random seed
    beta = np.empty([len(x_temp[0]), 1])
    for i in range(len(x_temp[0])):
        beta[i][0] = random.random()

    m = y_train.size

    # Start time
    t_start = perf_counter()

    # Batch gradient descent
    for i in range(nIteration):
        y_hat = np.dot(x_temp, beta)
        beta = beta - learning_rate * (1.0 / m) * np.dot(x_temp.T, y_hat - y_train)

    # End time
    t_stop = perf_counter()
    # Total time
    cpu_time = t_stop - t_start

    # x_temp = X_test_scaled
    x_temp = np.c_[np.ones((len(np.array(X_test_scaled)), 1)), np.array(X_test_scaled)]  # m+1

    # Prediction
    y_pred = x_temp.dot(beta)

    # Some error checking for RMSE
    if np.array(y_pred).shape == np.array(y_test).shape:
        RMSE = np.sqrt(np.mean((np.array(y_pred) - np.array(y_test)) ** 2))



    return (beta, y_pred, RMSE, cpu_time)
