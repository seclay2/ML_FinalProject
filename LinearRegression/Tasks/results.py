# print RESULTS


def results(self, y_test, y_pred, RMSE):
    import matplotlib.pyplot as plt

    # print RMSE
    print("RMSE: ", RMSE)

    # plot results
    import matplotlib.pyplot as plt
    plt.title('Actual vs. Predicted Acceleration')
    plt.ylabel('Acceleration $10 m/s^2$')
    plt.xlabel("Sample")
    plt.legend(['Predicted', 'Actual'], loc='upper left')
    plt.plot(y_test.reset_index(drop=True))    # y_test data
    plt.plot(y_pred)                           # y_pred data
    plt.legend(["Actual", "Predicted"])
    plt.show()
