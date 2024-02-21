

from src.data import apply_normalization, compute_stats, DHSFeatherData
from torch.utils.data import DataLoader


dataset = DHSFeatherData("data/filtered_dataset.ftr", label_columns=(11, 15),
                         dna_sequence_column='sequence', 
                         feature_columns=['total_signal', 'component'])




dataset = apply_normalization(dataset, compute_stats(dataset, columns=['total_signal']))
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for t in loader:
    print(t)
    break


print(dataset[0])
total_signal = [t['total_signal'] for t in dataset]
import numpy as np
print(np.mean(total_signal))
print(np.std(total_signal))