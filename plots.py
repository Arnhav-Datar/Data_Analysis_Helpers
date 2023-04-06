import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import pinv_extended
from typing import List, Dict, Union, Tuple, Optional
import os
import os.path as osp
sns.set(style="darkgrid")

class Plot:

    def __init__(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        plot_folder: str = "",
        plot_format: str = 'png',
        feature_descriptions: Optional[Dict[str, str]] = None,
    ):
        
        self.labels = labels
        self.features = features
        self.plot_folder = plot_folder
        self.plot_format = plot_format
        self.feature_descriptions = feature_descriptions
        if self.labels is not None:
            self.df = pd.concat([self.labels, self.features], axis=1)
        else:
            self.df = self.features
            
        if(self.plot_folder == ""):
            self.plot_folder = "./plots/"
            if(not osp.exists(self.plot_folder)):
                os.makedirs(self.plot_folder)

    def plot_single(
        self, 
        x: str,
        y: str,
        title: str = "",
        axis_labels: Tuple[str, str] = ["", ""],
        save: bool = True,
        height: int = 7,
        filename: str = "",
    ):
        
        g = sns.jointplot(
            x=x,
            y=y,
            data=self.df,
            kind="reg",
            truncate=False,
            color="m",
            height= height,
            joint_kws={'line_kws':{'color':'red'}}    
        )

        if axis_labels[0] == "":
            axis_labels[0] = x
        if axis_labels[1] == "":
            axis_labels[1] = y
        if title == "":
            title = f"{axis_labels[1]} vs {axis_labels[0]}"
        
        g.set_axis_labels(axis_labels[0], axis_labels[1], fontsize=16)
        g.fig.suptitle("Jointplot of " + title, fontsize=16)
        # g.ax_joint.collections[0].set_alpha(0)
        # g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        
        if save:
            plt.savefig(osp.join(self.plot_folder, filename), format=self.plot_format)
        else:
            plt.show()

    def plot_all(
        self,
        title: str = "",
        axis_labels: Tuple[str, str] = ["", ""],
        save: bool = True,
        height: int = 7,
        filename: str = "",
    ):
        g = sns.pairplot(self.df, kind="reg", height=height)
        g.fig.suptitle("Pairplot of " + title, fontsize=16)
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        
        if save:
            plt.savefig(osp.join(self.plot_folder, filename), format=self.plot_format)
        else:
            plt.show()
        
        
        