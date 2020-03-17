def Q_10(self, X_train_scaled, X_test_scaled, y_train, y_test):
    # Task 10: Given the (X_train, y_train) pairs denoting input matrix and output vector respectively,
    # Fit a linear regression model using the close-form solution you learned in class to obtain
    # the coefficients, beta's, as a numpy array of m+1 values (Please recall class lecture).
    # Then using the computed beta values, predict the test samples provided in the "X_test_scaled"
    # argument, and let's call your prediction "y_pred".
    # Compute Root Mean Squared Error (RMSE) of your prediction.
    # Finally, return the beta vector, y_pred, RMSE as a tuple.
    # PLEASE DO NOT USE ANY LIBRARY FUNCTION THAT DOES THE LINEAR REGRESSION.

    beta = []
    y_pred = []
    RMSE = -1

    ## YOUR CODE HERE ###
    import numpy as np

    # Add a column of 1's for the bias
    x_temp = np.c_[np.ones((len(np.array(X_train_scaled)), 1)), np.array(X_train_scaled)]  # m+1

    # Calculate beta for doing the prediction
    beta = ((np.linalg.inv(x_temp.transpose().dot(x_temp))).dot(x_temp.transpose())).dot(y_train)

    # Bias
    x_temp = np.c_[np.ones((len(np.array(X_test_scaled)), 1)), np.array(X_test_scaled)]  # m+1

    # Prediction
    y_pred = x_temp.dot(beta)

    # Some error checking for RMSE
    if np.array(y_pred).shape == np.array(y_test).shape:
        RMSE = np.sqrt(np.mean((np.array(y_pred) - np.array(y_test)) ** 2))

    return (beta, y_pred, RMSE)