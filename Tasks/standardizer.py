def scaler(self, X_train, X_test, y_train, y_test):
    # Given the 4 splits denoting the training and test dataset,
    # Applies standardization (i.e., normalization) scaling on the training dataset (X_train).
    # Then scales the test dataset based on the metrics obtained when scaling the training dataset.

    # Assuming standard deviation is never 0
    X_train_scaled = (X_train - X_train.mean()) / X_train.std()

    X_test_scaled = (X_test - X_train.mean()) / X_train.std()

    return X_train_scaled, X_test_scaled, y_train, y_test
