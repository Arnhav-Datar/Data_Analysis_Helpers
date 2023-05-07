import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import os.path as osp
import warnings
from sdf import SDF
sns.set(style="darkgrid")


class Plot(SDF):

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
        g.fig.subplots_adjust(top=0.95)
        self.savefig_or_show(save, filename)

    def plot_all(
        self,
        title: str = "",
        save: bool = True,
        height: int = 7,
        filename: str = "",
    ):
        '''
        Plots all features against each other in a pairplot
        '''

        g = sns.pairplot(self.df, kind="reg", height=height)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=2)
        
        if title == "":
            title = self.dataframe_name

        g.fig.suptitle("Pairplot of " + title, fontsize=16)
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        self.savefig_or_show(save, filename)
               
    def plot_heatmap_correlations(
        self,
        annot: bool = True,
        save: bool = True,
        filename: str = "",
        min_max: Tuple[float, float] = (-1, 1),
    ):
        
        '''
        Get a heatmap for the correlations between features
        '''

        corr = self.df.corr()
        f, ax = plt.subplots(figsize=(9, 6))
        g = sns.heatmap(
            corr,
            annot=annot,
            linewidths=.5,
            ax=ax,
            vmin = min_max[0],
            vmax = min_max[1],
        )
        self.savefig_or_show(save, filename)

    def plot_timeseries(
        self, 
        features: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "",
    ):
        '''
        Plot a timeseries
        '''
        if features is None:
            features = self.df.columns
        g = sns.lineplot(data=self.df[features], palette="tab10", linewidth=2.5)
        self.savefig_or_show(save, filename)

    def plot_residuals(
        self,
        feature_orders: Dict[str, int] = {},
        sub_folder: str = "", 
    ):
        '''
        Plot the residuals of the model
        '''

        if feature_orders == {}:
            feature_orders = {feature: 1 for feature in self.features.columns}

        if sub_folder == "":
            sub_folder = "residuals"

        self.safe_mkdir(osp.join(self.plot_folder, sub_folder))

        for feature in feature_orders:
            g = sns.residplot(
                data=self.df,
                x=feature,
                y=self.df.columns[-1],
                lowess=True,
                line_kws=dict(color="r"),
            )
            plt.title(f"Residuals of {feature} vs {self.labels.name}")
            self.savefig_or_show(True, osp.join(sub_folder, f"residuals_{feature}.png"))
        