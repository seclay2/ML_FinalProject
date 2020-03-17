def Q_01(self):
    #Task 1: Compute mean, stdev, min, max, 25% percentile, median and 75% percentile of BWEIGHT target variable
    # in the full_dataset, and return the computed values as a numpy array containing these 7 numbers (respectively).


    ## YOUR CODE HERE ##

    # MEAN
    mean = self.full_dataset['BWEIGHT'].mean()

    # STDDEV
    stddev = self.full_dataset['BWEIGHT'].std()

    # MIN
    min = self.full_dataset['BWEIGHT'].min()

    # MAX
    max = self.full_dataset['BWEIGHT'].max()

    # 25th PERCENTILE
    twenty_fifth_percentile = self.full_dataset['BWEIGHT'].quantile(0.25)

    # MEDIAN
    median = self.full_dataset['BWEIGHT'].median()

    # 75th PERCENTILE
    seventy_fifth_percentile = self.full_dataset['BWEIGHT'].quantile(0.75)

    output = [mean, stddev, min, max, twenty_fifth_percentile, median, seventy_fifth_percentile]

    return output