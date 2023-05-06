import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import os.path as osp
import warnings
sns.set(style="darkgrid")

class Plot:

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

        # Fit a linear model
        # lr = LinearRegression()
        # lr.fit(self.features, self.labels)
        # self.lr = lr
        # self.simple_predictions = lr.predict(self.features)

        if self.labels is not None:
            self.df = pd.concat([self.features, self.labels], axis=1)
        else:
            self.df = self.features
            
        if(self.plot_folder == ""):
            self.plot_folder = "./plots/"
            self.safe_mkdir(self.plot_folder)

    def safe_mkdir(self, path):
        if(not osp.exists(path)):
            os.makedirs(path)

    def savefig_or_show(self, save: bool, filename: str):
        if save:
            plt.savefig(osp.join(self.plot_folder, filename), format=self.plot_format)
        else:
            plt.show()

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
            print(feature)

            g = sns.residplot(
                data=self.df[list(feature_orders.keys())],
                x=feature,
                y=self.labels,
                lowess=True,
                line_kws=dict(color="r"),
            )
            plt.title(f"Residuals of {feature} vs {self.labels.name}")
            self.savefig_or_show(True, osp.join(sub_folder, f"residuals_{feature}.png"))
        