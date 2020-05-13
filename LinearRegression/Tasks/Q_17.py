def Q_17(self, dataset):
    import matplotlib.pyplot as plt

    # Split into features and target
    X, y = self.feature_target_split(dataset, ' Logged_Acceleration')

    # Split into training and test data
    X_train = X[:-3500]
    X_test = X[-3500:]
    y_train = y[:-3500]
    y_test = y[-3500:]

    # --- Scale all the data the same way ---
    X_train_scaled, X_test_scaled, y_train, y_test = self.scaler(X_train, X_test, y_train, y_test)

    # Plot config / plotting y actual
    plt.plot(y_test.reset_index(drop=True))
    plt.title('Actual vs. Predicted Acceleration')
    plt.ylabel('Acceleration $10 m/s^2$')
    plt.xlabel('Time $s$')

    # --- To do the actual prediction on the Judge Dataset Using Gradient Descent ---
    #minibatch = self.Q_13(X_train_scaled, X_test_scaled, y_train, y_test)
    #beta, y_pred, RMSE, cpu_time = minibatch

    #stochastic gradient descent regression
    y_pred, RMSE = self.SGD(X_train_scaled, X_test_scaled, y_train, y_test)
    print("\n\n_____________RMSE: ", RMSE)
    plt.plot(y_pred)
    plt.legend(["Actual", "Predicted"])
    plt.show()

    # --- Uncomment this to perform the prediction using standard Linear Regression ---
    # beta, y_pred, RMSE = self.Q_10(X_train, X_test_judge, y_train, y_test)


    return y_pred