from sdf import SDF
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
import numpy as np
from regression import RegressionAnalysis
sns.set(style="darkgrid")

class FeatureSelection(SDF):

    def lasso_regression(self, folds = 5, best_features=20, repeats=5):

        # Lasso Regression
        model = LassoCV(cv=folds, random_state=0, max_iter=10000)
        model.fit(self.df.drop(columns=[self.label_column]), self.df[self.label_column])
        coef = pd.Series(model.coef_, index = self.df.drop(columns=[self.label_column]).columns)

        # Printing the results for the best alpha       
        print("Best alpha using built-in LassoCV: %f" % model.alpha_)
        print("Best score using built-in LassoCV: %f" % model.score(self.df.drop(columns=[self.label_column]), self.df[self.label_column]))
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

        # Print the selected features and their coefficients
        print("Selected features: ")
        for i in range(len(coef)):
            if coef[i] != 0:
                print(self.df.drop(columns=[self.label_column]).columns[i], coef[i])

        # Analyze the regression
        print(self.get_model_statistics(model).summary())

        # PLotting the feature coefficients
        figsize=(10, 10)
        cv_model = cross_validate(
            model, self.features, self.labels, cv=RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=0),
            return_estimator=True, n_jobs=5
        )
        coefs = pd.DataFrame(
            [model.coef_ for model in cv_model['estimator']],
            columns=self.features.columns
        )
        
        coefs.loc[len(coefs.index)] = coefs.mean().abs()
        coefs = coefs.sort_values(by = len(coefs.index) - 1, axis=1, ascending=False)
        coefs = coefs.drop(coefs.tail(1).index)

        plt.figure(figsize=(9, 7))
        sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5)
        plt.axvline(x=0, color='.5')
        plt.xlabel('Coefficient importance')
        plt.title('Coefficient importance and its variability')
        self.savefig_or_show(True, "lasso_features_best_alpha.png")

        # Plotting the feature coefficients for different alphas
        alphas = np.linspace(0.01,500,20)
        lasso = Lasso(max_iter=10000)
        coefs = []

        for a in alphas:
            lasso.set_params(alpha=a)
            lasso.fit(self.df.drop(columns=[self.label_column]), self.df[self.label_column])
            coefs.append(lasso.coef_)

        coefs = pd.DataFrame(coefs, index = alphas, columns = self.df.drop(columns=[self.label_column]).columns)
        ax = plt.gca()
        sns.lineplot(data= coefs, palette="tab10", linewidth=2.5)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel('alpha')
        plt.ylabel('Standardized Coefficients')
        plt.title('Lasso coefficients as a function of alpha')
        self.savefig_or_show(True, "lasso_features_alpha.png")