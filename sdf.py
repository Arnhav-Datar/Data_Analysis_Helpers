import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import os.path as osp
import warnings

class SDF:

    def __init__(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        plot_folder: str = "",
        plot_format: str = 'png',
        feature_descriptions: Optional[Dict[str, str]] = None,
        dataframe_name: str = "dataframe",
    ):
        
        self.labels = labels
        self.features = features

        if self.labels is None:
            warnings.warn("No label column provided. Taking last column as label!")
            self.labels = self.features[self.features.columns[-1]]
            self.features = self.features.drop([self.features.columns[-1]], axis = 1)

        self.plot_folder = plot_folder
        self.plot_format = plot_format
        self.feature_descriptions = feature_descriptions
        self.dataframe_name = dataframe_name

        if self.labels is not None:
            self.df = pd.concat([self.features, self.labels], axis=1)
        else:
            self.df = self.features
            
        if(self.plot_folder == ""):
            self.plot_folder = "./plots/"
            self.safe_mkdir(self.plot_folder)

        self.label_column = self.df.columns[-1]

    def safe_mkdir(self, path):
        if(not osp.exists(path)):
            os.makedirs(path)

    def savefig_or_show(self, save: bool, filename: str):
        if save:
            plt.savefig(osp.join(self.plot_folder, filename), format=self.plot_format)
            plt.clf()
        else:
            plt.show()