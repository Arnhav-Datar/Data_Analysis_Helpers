from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn, statsmodels
import numpy as np
from sdf import SDF

class RegressionAnalysis(SDF):

    def get_model_statistics(self, model):
        
        is_statsmodels = False
        is_sklearn = False
        has_intercept = False
        X = self.features
        y = self.labels
        
        # check for accepted linear models
        if type(model) in [sklearn.linear_model._base.LinearRegression,
                        sklearn.linear_model._ridge.Ridge,
                        sklearn.linear_model._ridge.RidgeCV,
                        sklearn.linear_model._coordinate_descent.Lasso,
                        sklearn.linear_model._coordinate_descent.LassoCV,
                        sklearn.linear_model._coordinate_descent.ElasticNet,
                        sklearn.linear_model._coordinate_descent.ElasticNetCV,
                        sklearn.linear_model.HuberRegressor,
                        ]:
            is_sklearn = True
        elif type(model) in [statsmodels.regression.linear_model.OLS, 
                            statsmodels.base.elastic_net.RegularizedResults,
                            ]:
            is_statsmodels = True
        else:
            print("Only linear models are supported!")
            return None
        
        if is_statsmodels and all(np.array(X)[:,0]==1):
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
        pinv_wexog,_ = pinv_extended(x)
        normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
        return sm.regression.linear_model.OLSResults(olsModel, model_params, normalized_cov_params)
    