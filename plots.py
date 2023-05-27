import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import os.path as osp
import statsmodels.regression.linear_model as sm
import statsmodels.api as sm1
from scipy import stats
import warnings
from sdf import SDF
import pylab
import numpy as np

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
        filename: str = "jointplot",
    ):

        g = sns.jointplot(
            x=x,
            y=y,
            data=self.df,
            kind="reg",
            truncate=False,
            color="m",
            height=height,
            joint_kws={"line_kws": {"color": "red"}},
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
        self.savefig_or_show(save, filename + ".png")

    def plot_all(
        self,
        title: str = "",
        save: bool = True,
        height: int = 7,
        filename: str = "pairplot",
    ):
        """
        Plots all features against each other in a pairplot
        """

        g = sns.pairplot(self.df, kind="reg", height=height)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=2)

        if title == "":
            title = self.dataframe_name

        g.fig.suptitle("Pairplot of " + title, fontsize=16)
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        self.savefig_or_show(save, filename + ".png")

    def plot_heatmap_correlations(
        self,
        annot: bool = True,
        save: bool = True,
        filename: str = "correlations",
        min_max: Tuple[float, float] = (-1, 1),
        modes: List[str] = ["pearson", "kendall", "spearman", "beta"],
    ):

        """
        Get a heatmap for the correlations between features
        """

        corr = self.df.corr(numeric_only=True)
        ln = corr.shape[0]
        clms = corr.columns

        for md in modes:

            if md == "beta":
                
                for i in range(ln):
                    for j in range(ln):
                        col1, col2 = clms[i], clms[j]
                        cov = np.cov(self.df[col1], self.df[col2])
                        corr.iat[i, j] = cov[0, 1] / cov[0, 0]

            else:
                corr = self.df.corr(method = md, numeric_only=True)
            
            
            _, ax = plt.subplots(figsize=(9, 6))
            g = sns.heatmap(
                corr,
                annot=annot,
                linewidths=0.5,
                ax=ax,
                vmin=min_max[0],
                vmax=min_max[1],
            )

            g.set_xticklabels(g.get_yticklabels(), rotation=20, horizontalalignment="right")
            g.set_yticklabels(g.get_yticklabels(), rotation=20, horizontalalignment="right")

            plt.title(f"Correlations ({md})")
            self.savefig_or_show(save, filename + "_" + md + ".png")

    def plot_timeseries(
        self,
        features: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "time_series",
    ):
        """
        Plot a timeseries
        """
        if features is None:
            features = self.df.columns
        g = sns.lineplot(data=self.df[features], palette="tab10", linewidth=2.5)
        self.savefig_or_show(save, filename + ".png")

    def plot_residuals(
        self,
        feature_orders: Dict[str, int] = {},
        sub_folder: str = "residuals",
    ):
        """
        Plot the residuals of the model
        """

        if feature_orders == {}:
            feature_orders = {feature: 1 for feature in self.features.columns}

        self.safe_mkdir(osp.join(self.plot_folder, sub_folder))

        for feature in feature_orders:
            g = sns.residplot(
                data=self.df,
                x=feature,
                y=self.df.columns[-1],
                lowess=True,
                line_kws=dict(color="r"),
                order=feature_orders[feature],
            )
            plt.title(f"Residuals of {feature} vs {self.labels.name}")
            self.savefig_or_show(True, osp.join(sub_folder, f"residuals_{feature}.png"))

        # Get the qq plots of the model

        for feature in feature_orders:
            # Doing a polynomial regression
            formula = f"{self.label_column} ~ {feature}**{feature_orders[feature]}"
            print(formula)
            reg = sm.OLS.from_formula(formula, data=self.df).fit()
            residuals = np.asarray(reg.resid.tolist())
            residuals /= np.std(residuals, ddof=1)
            sm1.qqplot(residuals, line="45")
            plt.title(formula)
            self.savefig_or_show(True, osp.join(sub_folder, f"qq_{feature}.png"))
