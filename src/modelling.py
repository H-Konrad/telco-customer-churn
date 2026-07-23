from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TunedThresholdClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve

class ModelTraining:
    def __init__(
            self, 
            model, 
            data,
            target,
            numerical_features,
            binary_features,
            categorical_features,
            test_size = 0.2,
            random_state = 123
        ):
        self.model = model
        self.X = data.drop(columns = [target])
        self.y = data[target]
        self.numerical_features = numerical_features
        self.binary_features = binary_features
        self.categorical_features = categorical_features
        self.test_size = test_size
        self.random_state = random_state
        self.build_pipeline()
        self.split_data()


    def build_pipeline(self):
        self.preprocessor = ColumnTransformer(
            transformers = [
                (
                    "numerical",
                    StandardScaler(),
                    self.numerical_features
                ),
                (
                    "binary",
                    OrdinalEncoder(),
                    self.binary_features
                ),
                (
                    "categorical",
                    OneHotEncoder(),
                    self.categorical_features
                )
            ]
        )

        self.model = Pipeline(
            steps = [
                ("preprocessor", self.preprocessor),
                ("model", self.model)
            ]
        )


    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size = self.test_size,
            random_state = self.random_state,
            stratify = self.y
        )

        
    def train(self):
        self.model.fit(
            self.X_train,
            self.y_train
        )


    def optimise(
            self, 
            params, 
            scoring = "balanced_accuracy", 
            n_iter = 100, 
            cv = 5,
            random_state = 123
        ):
        self.randomised_search = RandomizedSearchCV(
            estimator = self.model,
            param_distributions = params,
            scoring = scoring,
            n_iter = n_iter,
            n_jobs = -1,
            cv = cv,
            random_state = random_state
        )

        self.randomised_search.fit(
            self.X_train,
            self.y_train
        )

        self.model = self.randomised_search.best_estimator_

        print(f"Best params: {self.randomised_search.best_params_}\n")


    def threshold(
            self, 
            scoring = "balanced_accuracy", 
            cv = 3, 
            random_state = 123
        ):
        self.threshold_model = TunedThresholdClassifierCV(
            estimator = self.model,
            scoring = scoring,
            n_jobs = -1,
            cv = cv,
            random_state = random_state
        )

        self.threshold_model.fit(
            self.X_train,
            self.y_train
        )

        self.model = self.threshold_model

        print(f"Threshold: {self.model.best_threshold_:.4f}\n")


    def evaluate(self, report = False):
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)

        print("TRAINING SET")
        print(f"Accuracy:{accuracy_score(self.y_train, train_predictions) * 100.0: .2f}%")
        print(f"Confusion Matrix:\n{confusion_matrix(self.y_train, train_predictions)}")

        print("\nTESTING SET")
        print(f"Accuracy:{accuracy_score(self.y_test, test_predictions) * 100.0: .2f}%")
        print(f"Confusion Matrix:\n{confusion_matrix(self.y_test, test_predictions)}")
        if report:
            print(f"Classification Report:\n{classification_report(self.y_test, test_predictions)}")
