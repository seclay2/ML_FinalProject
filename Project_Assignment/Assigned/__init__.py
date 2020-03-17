class Assigned:
    def __init__(self, full_dataset_filename, judge_filename):
        self.full_dataset_filename =full_dataset_filename
        self.judge_filename = judge_filename

        import pandas as pd
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 10)
        # First load the dataset into pandas dataframe
        self.full_dataset = pd.read_csv(full_dataset_filename,delimiter=',')
        self.judge_dataset = pd.read_csv(judge_filename,delimiter=',')




    #Imported methods
    # from .tasks.Q_01 import Q_01
    from .tasks.Q_02 import Q_02
    from .tasks.Q_03 import Q_03
    from .tasks.Q_04 import Q_04
    from .tasks.Q_05 import Q_05
    from .tasks.Q_06 import Q_06
    from .tasks.Q_07 import Q_07
    from .tasks.Q_08 import Q_08
    from .tasks.Q_09 import Q_09
    from .tasks.Q_10 import Q_10
    from .tasks.Q_11 import Q_11
    from .tasks.Q_12 import Q_12
    from .tasks.Q_13 import Q_13
    from .tasks.Q_14 import Q_14
    from .tasks.Q_15 import Q_15
    from .tasks.Q_16 import Q_16
    from .tasks.Q_17 import Q_17
    from .tasks.EDA import EDA




















