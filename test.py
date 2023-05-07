import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Tuple, Optional, Union
from plots import Plot
from feature_selection import FeatureSelection

sns.set_style('darkgrid')

penguins = sns.load_dataset("penguins")
# pl = Plot(penguins[['species', 'island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g']], plot_folder="./plots/")
# pl.plot_single(x="bill_length_mm", y="bill_depth_mm", title="Penguin Bill Length vs. Depth", axis_labels=("Bill Length (mm)", "Bill Depth (mm)"), save=True, height=7, filename="penguin_bill_length_vs_depth.png")
# pl.plot_all(save=True, height=7, filename="penguin_all.png")
# pl.plot_heatmap_correlations(save=True, filename="penguin_heatmap_correlations.png")
# pl.plot_residuals({'bill_length_mm':2, 'bill_depth_mm':2,  'flipper_length_mm':2})

# sample N points from normal distribution
N = 1000
y = np.random.normal(0, 1, N)
x1 = y + np.random.normal(0, 1, N)
x2 = - y + np.random.normal(0, 1, N)
x3 = y + np.random.normal(0, 1, N)
x4 = y + np.random.normal(0, 1, N)
x5 = y + np.random.normal(0, 1, N)

# create dataframe with 15 junk columns
df = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4, 'x5':x5})
for i in range(15):
    df[f'junk{i}'] = np.random.normal(0, 1, N)
df['y'] = y

fs = FeatureSelection(df, plot_folder="./plots/")
fs.lasso_regression()


