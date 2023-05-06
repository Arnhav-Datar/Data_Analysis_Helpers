import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Tuple, Optional, Union
from plots import Plot

sns.set_style('darkgrid')

penguins = sns.load_dataset("penguins")
pl = Plot(penguins, plot_folder="./plots/")
# pl.plot_single(x="bill_length_mm", y="bill_depth_mm", title="Penguin Bill Length vs. Depth", axis_labels=("Bill Length (mm)", "Bill Depth (mm)"), save=True, height=7, filename="penguin_bill_length_vs_depth.png")
# pl.plot_all(save=True, height=7, filename="penguin_all.png")
# pl.plot_heatmap_correlations(save=True, filename="penguin_heatmap_correlations.png")
pl.plot_residuals({'bill_length_mm':1, 'bill_depth_mm':1,  'flipper_length_mm':1})


