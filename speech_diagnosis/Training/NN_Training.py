import pandas as pd

class NN_Training:

    def __int__(self):

        data = pd.read_csv('train_data.txt', sep=",", header=None)

        columns = ["Subject id",
                   "Jitter (local)", "Jitter (local, absolute)", "Jitter (rap)", "Jitter (ppq5)",
                   "Jitter (ddp)",

                   "Shimmer (local)", "Shimmer (local, dB)", "Shimmer (apq3)", "Shimmer (apq5)",
                   "Shimmer (apq11)", "Shimmer (dda)",

                   "AC", "NTH", "HTN",

                   "Median pitch", "Mean pitch",
                   "Standard deviation", "Minimum pitch", "Maximum pitch",

                   "Number of pulses", "Number of periods",
                   "Mean period", "Standard deviation of period",

                   "Fraction of locally unvoiced frames",
                   "Number of voice breaks",
                   "Degree of voice breaks",

                   "UPDRS", "class information"]


