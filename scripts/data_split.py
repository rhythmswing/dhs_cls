
import numpy as np
import pandas as pd

np.random.seed(42)


master_data = pd.read_feather("data/master_dataset.ftr")
bio_sample_labels = master_data.iloc[:, 11:]
