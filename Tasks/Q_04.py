def Q_04(self, full_dataset_all_numeric):
    # Task 4: Given a full_dataset (Pandas dataframe) where all the categorical variables are
    # already replaced with numeric values, return a list of top 20 highly correlated variables
    # (with respect to the target variable) as a Pandas dataframe with 2 columns {variable,corr_score}.
    # The corr_score between a variable x and the target variable y needs to be computed using the
    # Pearson Correlation Coefficient.

    import pandas as pd

    ## YOUR CODE HERE ##
    corr_dataset = full_dataset_all_numeric

    # Perform the pearson correlation
    corr_dataset = corr_dataset.corr(method='pearson')
    corr_dataset = corr_dataset[['BWEIGHT']]

    # Find the strongest correlations
    corr_dataset = corr_dataset.reindex(corr_dataset.BWEIGHT.abs().sort_values(ascending=False).index)

    # Drop BWEIGHT (as it will have a correlation of 1)
    corr_dataset = corr_dataset.head(21).drop(['BWEIGHT']).reset_index()

    # Create the column headings
    corr_dataset.columns = ['variable_name', 'corr_score']


    return corr_dataset


