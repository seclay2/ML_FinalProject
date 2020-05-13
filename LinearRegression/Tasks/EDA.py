def EDA(self, dataset):
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("---HEAD---\n\n", dataset.head())
    print("\n---SHAPE---\n\n", dataset.shape)
    print("\n---COLUMN VALUE INFO---\n\n", dataset.info())
    print("\n---COLUMN DETAILS---\n\n", dataset.describe().round(3))

    # Correlation
    print("\n---CORRELATIONS---\n")
    corr = dataset.corr()
    sns.heatmap(corr, annot=True, fmt='.3g', vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
    plt.show()