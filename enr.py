from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sdf import SDF

class ElasticNetRegressor(SDF):

    def __init__(
        self,
        df: pd.DataFrame,
        label_cols: List[str],
        feature_cols: List[str] = None,
        plot_folder: str = "",
        plot_format: str = "png",
        feature_descriptions: Dict[str, str] = None,
    ):

        super().__init__(
            df,
            label_cols,
            feature_cols,
            plot_folder,
            plot_format,
            feature_descriptions,
        )

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.scaled_features = None

    def scale_features(
        self,
        scaler: StandardScaler = None,
    ) -> pd.DataFrame:
        '''
        Scale the features

        Returns:
            scaled features
        '''

        # save the names of the features
        self.feature_names = self.features.columns

        # scale the features
        if scaler is None:
            self.scaled_features = self.scaler.transform(self.features.to_numpy())
        else:
            self.scaled_features = scaler.transform(self.features.to_numpy())

        # convert the scaled features to a dataframe
        return pd.DataFrame(self.scaled_features, columns=self.feature_names)

    def train(
        self,
        scale: bool = True,
        label_col: str = None,
        n_folds: int = 7,
        verbose_level: int = 1,
    ) -> None:
        '''
        Train the model
        scale: whether to scale the features
        print_summary: whether to print the model summary
        label_col: label column name
        '''

        if label_col is None:
            label_col = self.label_cols[0]


        if verbose_level > 0:
            print("====================================")
            print("Training the model...")
            print("====================================")

            print("We are using the following features")
            print(self.features.columns)

            print("to predict")
            print(label_col)

        
        y = self.labels[label_col]
        X = self.features

        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            self.scale_features()
            X = self.scaled_features
        
        self.model = sklearn.linear_model.ElasticNetCV(
            cv = TimeSeriesSplit(
                n_splits = n_folds,
                test_size= len(X) // (2 * n_folds),
                gap = 2,
            ),
            fit_intercept = True,
        )

        self.model.fit(X, y)

        predictions = self.model.predict(X)

        if verbose_level > 0:
            
            print("The training R2 score is ", sklearn.metrics.r2_score(y, predictions))

            if verbose_level >= 1 and verbose_level < 2:
                print("The model coefficients are ")
                coefs = pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "coefficient": self.model.coef_,
                    }
                )
                print("The intercept is ", self.model.intercept_)

                print(coefs)

            elif verbose_level >= 2:
                
                print(X.shape)
                print(self.features.shape)

                print(
                    self.get_model_statistics(
                        self.model, X, label_col
                    ).summary()
                )



    def predict(
        self,
        sdf : SDF,
    ):

        '''
        Predict the labels
        '''

        sdf.scale_features(self.scaler)
        X = sdf.scaled_features

        return self.model.predict(X)
