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
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        plot_folder: str = "",
        plot_format: str = "png",
        feature_descriptions: Optional[Dict[str, str]] = None,
        dataframe_name: str = "dataframe",
    ):

        self.labels = labels
        self.features = features

        if self.labels is None:
            warnings.warn("No label column provided. Taking last column as label!")
            self.labels = self.features[self.features.columns[-1]]
            self.features = self.features.drop([self.features.columns[-1]], axis=1)

        self.plot_folder = plot_folder
        self.plot_format = plot_format
        self.feature_descriptions = feature_descriptions
        self.dataframe_name = dataframe_name

        if self.labels is not None:
            self.df = pd.concat([self.features, self.labels], axis=1)
        else:
            self.df = self.features

        if self.plot_folder == "":
            self.plot_folder = "./plots/"
            self.safe_mkdir(self.plot_folder)

        self.label_column = self.df.columns[-1]

    def safe_mkdir(self, path):
        if not osp.exists(path):
            os.makedirs(path)

    def savefig_or_show(self, save: bool, filename: str):
        if save:
            plt.savefig(osp.join(self.plot_folder, filename), format=self.plot_format)
            plt.clf()
        else:
            plt.show()

    def get_model_statistics(self, model):

        is_statsmodels = False
        is_sklearn = False
        has_intercept = False
        X = self.features
        y = self.labels

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
