import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
import os.path as osp
from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn, statsmodels
import numpy as np
import warnings


class SDF:
    def __init__(
        self,
        df: pd.DataFrame,
        label_cols: List[str],
        feature_cols: List[str] = None,
        plot_folder: str = "",
        plot_format: str = "png",
        feature_descriptions: Optional[Dict[str, str]] = None,
    ):

        '''
        Initialize the SDF object
        df: pandas dataframe
        label_cols: list of column names of the labels
        feature_cols: list of column names of the features
        plot_folder: folder to save plots
        plot_format: format of the plots
        feature_descriptions: dictionary of feature descriptions
        '''

        self.df = df

        self.label_cols = label_cols
        self.labels = self.df[label_cols]

        if feature_cols is None:
            self.feature_cols = self.df.columns.drop(label_cols)
            self.features = self.df.drop(label_cols, axis=1)
        else:
            self.feature_cols = feature_cols
            self.features = self.df[feature_cols]

        self.plot_folder = plot_folder
        self.plot_format = plot_format
        self.feature_descriptions = feature_descriptions

        if self.feature_descriptions is not None:
            self.print_feature_descriptions()

        if self.plot_folder == "":
            self.plot_folder = "./plots/"
            self.safe_mkdir(self.plot_folder)

        self.label_column = self.df.columns[-1]

    def safe_mkdir(self, path):
        '''
        Create a directory if it does not exist
        '''
        if not osp.exists(path):
            os.makedirs(path)

    def print_feature_descriptions(
        self,
    ):
        '''
        Print the feature and label descriptions
        '''

        print("The features are ")
        for feature in self.feature_cols:
            print(feature + ": " + self.feature_descriptions[feature])

        print("====================================")
        
        print("The labels are ")
        for label in self.label_cols:
            print(label + ": " + self.feature_descriptions[label])
            

    def savefig_or_show(
        self,
        filename: str = None,
    ):
        '''
        Save the figure or show it
        '''
        if filename is not None:
            plt.savefig(osp.join(self.plot_folder, filename), format=self.plot_format)
            plt.clf()
        else:
            plt.show()

    def get_model_statistics(
        self,
        model,
        features: pd.DataFrame = None,
        label: str = None,
    ):

        '''
        Get the model statistics
        model: linear model
        '''

        is_statsmodels = False
        is_sklearn = False
        has_intercept = False
        X = features

        if label is None:
            y = self.labels[self.label_cols[0]]
        else:
            y = self.labels[label]

        y = list(y)

        # check for accepted linear models
        if type(model) in [
            sklearn.linear_model._base.LinearRegression,
            sklearn.linear_model._ridge.Ridge,
            sklearn.linear_model._ridge.RidgeCV,
            sklearn.linear_model._coordinate_descent.Lasso,
            sklearn.linear_model._coordinate_descent.LassoCV,
            sklearn.linear_model._coordinate_descent.ElasticNet,
            sklearn.linear_model._coordinate_descent.ElasticNetCV,
            sklearn.linear_model.HuberRegressor,
        ]:
            is_sklearn = True
        elif type(model) in [
            statsmodels.regression.linear_model.OLS,
            statsmodels.base.elastic_net.RegularizedResults,
        ]:
            is_statsmodels = True
        else:
            print("Only linear models are supported!")
            return None

        if is_statsmodels and all(np.array(X)[:, 0] == 1):
            has_intercept = True
        elif is_sklearn and model.intercept_:
            has_intercept = True

        if is_statsmodels:
            x = X
            model_params = model.params
        else:
            if has_intercept:
                x = sm.add_constant(X)
                model_params = np.hstack([np.array([model.intercept_]), model.coef_])
            else:
                x = X
                model_params = model.coef_

        olsModel = sm.OLS(y, x)
        pinv_wexog, _ = pinv_extended(x)
        normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
        return sm.regression.linear_model.OLSResults(
            olsModel, model_params, normalized_cov_params
        )

    def get_correlations(self):
        '''
        Get the correlations
        '''
        
        cr = self.df.corr()[self.label_cols]
        cr = cr.drop(self.label_cols, axis=0)
        sns.heatmap(
            cr,
            annot=True
        )
        self.savefig_or_show()

    def two_dimensions_plot(
        self,
        plot_func,  
        xsplit_col: str,
        ysplit_col: str = None,
    ):

        # get the unique values of the split columns
        xsplit_vals = self.df[xsplit_col].unique()
        ysplit_vals = self.df[ysplit_col].unique()

        # set the plot size
        plt.figure(figsize=(5 * len(xsplit_vals), 5 * len(ysplit_vals)))

        # Iterate over the unique values of the split columns
        for i, xsplit_val in enumerate(xsplit_vals):
            for j, ysplit_val in enumerate(ysplit_vals):
                # select the data with the unique values of the split columns
                df = self.df[
                    (self.df[xsplit_col] == xsplit_val)
                    & (self.df[ysplit_col] == ysplit_val)
                ]
                # plot the data
                plt.subplot(len(xsplit_vals), len(ysplit_vals), i * len(ysplit_vals) + j + 1)
                plot_func(df)
                plt.title(xsplit_col + ": " + str(xsplit_val) + ", " + ysplit_col + ": " + str(ysplit_val))

        plt.show()