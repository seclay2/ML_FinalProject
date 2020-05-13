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
    corr_dataset = corr_dataset[[' Logged_Acceleration']]

    # Find the strongest correlations
    corr_dataset = corr_dataset.reindex(corr_dataset.loc[' Logged_Acceleration'].abs().sort_values(ascending=False).index)

    # Drop logged_accel (as it will have a correlation of 1)
    corr_dataset = corr_dataset.head(7).drop([' Logged_Acceleration']).reset_index()

    # Create the column headings
    corr_dataset.columns = ['variable_name', 'corr_score']


    return corr_dataset


