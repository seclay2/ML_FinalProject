def min_max_scaler(self, X_train, X_test, y_train, y_test):
    # Given the 4 splits denoting the training and test dataset,
    # Applies standardization (i.e., normalization) scaling on the training dataset (X_train).
    # Then scales the test dataset based on the metrics obtained when scaling the training dataset.

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
