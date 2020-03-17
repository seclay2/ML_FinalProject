def EDA(self, dataset):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    print("---HEAD---\n", dataset.head())

    print("\n\n---SHAPE---\n", dataset.shape)

    print("\n\n---COLUMN VALUE INFO---\n", dataset.info())

    print("\n\n---COLUMN DETAILS---\n", dataset.describe().round(3))

    # Correlation
    print("\n\n---CORRELATIONS---\n")
    corr = dataset.corr()
    sns.heatmap(corr, annot=True, fmt='.3g', vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
    plt.show()

    print("\n\n---PREDICTIONS---")
    y_pred = self.Q_17(self.judge_dataset)
    print(y_pred)
    plt.plot(y_pred)
    plt.title('Acceleration Predictions')
    plt.ylabel('Acceleration')
    plt.xlabel('Time')
    plt.show()