from sdf import SDF
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV,
    ElasticNet,
    Lasso,
    LassoCV,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.model_selection import KFold, TimeSeriesSplit
import numpy as np
import lightgbm as lgb


sns.set(style="darkgrid")


class FeatureSelection(SDF):
    def elastic_net_regression(self, folds=5, best_features=20, repeats=5, l1_ratio=1):

        """
        Elastic Net Regression
        Parameters
        ----------
        folds : int, optional
            Number of folds for cross validation, by default 5
        best_features : int, optional
            Number of best features to plot, by default 20
        repeats : int, optional
            Number of repeats for cross validation, by default 5
        l1_ratio : int, optional
            Ratio of L1 to overall regularization, by default 1
        """

        if l1_ratio == 1:
            model = LassoCV(cv=folds, random_state=0, max_iter=10000)
        elif l1_ratio == 0:
            model = RidgeCV(cv=folds)
        else:
            model = ElasticNetCV(
                cv=folds, random_state=0, max_iter=10000, l1_ratio=l1_ratio
            )

        model.fit(self.df.drop(columns=[self.label_column]), self.df[self.label_column])
        coef = pd.Series(
            model.coef_, index=self.df.drop(columns=[self.label_column]).columns
        )

        # Printing the results for the best alpha
        print(
            "For ElasticNetCV with l1 ratio = "
            + str(l1_ratio)
            + ", the best alpha is "
            + str(model.alpha_)
        )
        print(
            "Best score using built-in ElasticNetCV: %f"
            % model.score(
                self.df.drop(columns=[self.label_column]), self.df[self.label_column]
            )
        )
        print(
            "ElasticNetCV picked "
            + str(sum(coef != 0))
            + " variables and eliminated the other "
            + str(sum(coef == 0))
            + " variables"
        )

        # Analyze the regression
        print(self.get_model_statistics(model).summary())

        # PLotting the feature coefficients
        figsize = (10, 10)
        cv_model = cross_validate(
            model,
            self.features,
            self.labels,
            cv=RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=0),
            return_estimator=True,
            n_jobs=5,
        )
        coefs = pd.DataFrame(
            [model.coef_ for model in cv_model["estimator"]],
            columns=self.features.columns,
        )

        coefs.loc[len(coefs.index)] = coefs.mean().abs()
        coefs = coefs.sort_values(by=len(coefs.index) - 1, axis=1, ascending=False)
        coefs = coefs.drop(coefs.tail(1).index)

        plt.figure(figsize=(9, 7))
        sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
        plt.axvline(x=0, color=".5")
        plt.xlabel("Coefficient importance")
        plt.title("Coefficient importance and its variability")
        self.savefig_or_show(True, "elastic_features_best_alpha.png")

        # Plotting the feature coefficients for different alphas
        alphas = np.linspace(0.01, 500, 20)
        lasso = ElasticNet(max_iter=10000, l1_ratio=l1_ratio)
        if l1_ratio == 1:
            lasso = Lasso(max_iter=10000)
        elif l1_ratio == 0:
            lasso = Ridge(max_iter=10000)

        coefs = []

        for a in alphas:
            lasso.set_params(alpha=a)
            lasso.fit(
                self.df.drop(columns=[self.label_column]), self.df[self.label_column]
            )
            coefs.append(lasso.coef_)

        coefs = pd.DataFrame(
            coefs,
            index=alphas,
            columns=self.df.drop(columns=[self.label_column]).columns,
        )
        ax = plt.gca()
        sns.lineplot(data=coefs, palette="tab10", linewidth=2.5)
        ax.set_xscale("log")
        plt.axis("tight")
        plt.xlabel("alpha")
        plt.ylabel("Standardized Coefficients")
        plt.title("ElasticNet coefficients as a function of alpha")
        self.savefig_or_show(True, "elastic_features_alpha.png")

    def lightgbm(
        self,
        n_folds: int = 5,
        fold_function=None,
        lr: float = 1e-2,
    ):
        """
        LightGBM model
        Parameters
        ----------
        folds : int, optional
            Number of folds for cross validation, by default 5
        fold_type : str, optional
            Type of cross validation, by default "kfold"
        lr : float, optional
            Learning rate, by default 1e-2
        """

        def plot_importance(cvbooster, figsize=(10, 10)):
            raw_importances = cvbooster.feature_importance(importance_type="gain")
            feature_name = cvbooster.boosters[0].feature_name()
            importance_df = pd.DataFrame(data=raw_importances, columns=feature_name)
            sorted_indices = (
                importance_df.mean(axis=0).sort_values(ascending=False).index
            )
            sorted_importance_df = importance_df.loc[:, sorted_indices]
            # plot top-n
            PLOT_TOP_N = 50
            plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
            _, ax = plt.subplots(figsize=figsize)
            ax.grid()
            ax.set_xscale("log")
            ax.set_ylabel("Feature")
            ax.set_xlabel("Importance")
            sns.boxplot(data=sorted_importance_df[plot_cols], orient="h", ax=ax)
            plt.tight_layout()
            self.savefig_or_show(True, "lightgbm_features_importance.png")

        def rmse(y_true, y_pred):
            return (np.sqrt(np.mean(np.square(y_true - y_pred))))

        def feval_RMSE(preds, train_data):
            labels = train_data.get_label()
            return "RMSE", round(rmse(y_true=labels, y_pred=preds), 5), False

        def make_folds(df, folds=5):
            if fold_function is None:
                spl = TimeSeriesSplit(
                    n_splits=folds, test_size=self.df.shape[0] // (2 * folds), gap=10
                )
                return spl
            else:
                return fold_function

        # params = {
        #     "objective": "regression"
        # }

        ds = lgb.Dataset(self.features, self.labels)

        folds = make_folds(self.df, folds=n_folds)
        hyper_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
        }

        ret = lgb.cv(
            hyper_params,
            train_set = ds,
            folds=folds,
            feval=feval_RMSE,
            return_cvbooster=True,
        )

        best_iteration = len(ret["RMSE-mean"])

        for i, (_, test_index) in enumerate(folds.split(self.df)):
            y_pred = (
                ret["cvbooster"]
                .boosters[i]
                .predict(self.features.iloc[test_index], num_iteration=best_iteration)
            )
            y_true = self.labels.iloc[test_index]
            print(f"# fold {i} RMSE: {rmse(y_true, y_pred)}")

        plot_importance(ret["cvbooster"], figsize=(10, 20))

    
