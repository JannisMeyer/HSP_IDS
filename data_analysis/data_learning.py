from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from . import data_processing as dp
from . import initial_data as i
# for later: k-fold cross-validation: split data into k subsets, train on k-1, test on 1
# -> do this k times and evaluate model
# Grid Search for hyperparameter tuning or Optuna


# region classifiers -------------------------------------------------------------------------------------------------------------------------------

# TODO: look at statistics after training, evtl. LSTM oder Transformer
def rfc(fvs, labels):
        x_train, x_test, y_train, y_test = train_test_split(fvs, labels, stratify=labels)
        rfc = RandomForestClassifier(verbose=False, n_jobs=28)

        grid = {'n_estimators':[100, 500, 1000],
                #'max_depth':[20],
                'max_depth':[3, 5, 7, 10, 20, 25],
                #'min_samples_leaf':[1]}
                'min_samples_leaf':[1, 2]}
        gs = GridSearchCV(estimator=rfc,
                          param_grid=grid,
                          scoring='accuracy',
                          cv=3,
                          return_train_score=True,
                          verbose=False)
        gs.fit(x_train, y_train.values.ravel())

        best_rfc = gs.best_estimator_
        predictions = best_rfc.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        feature_importances = best_rfc.feature_importances_

        return best_rfc, gs.best_params_, predictions, accuracy, feature_importances

def i_forest(fvs, labels):

    # TODO: transform labels to -1,1 here before usage
    # use GridSearch for Hyperparameter Tuning (n_estimators, max_samples, max_features)
    # for contamination: set to auto, train, test on various data sets and look at decision function values
    # use multiple processors
    # save IsolationForest as pkl-file

    # def f1_scorer(y, y_pred):

    #     # transform to array of either -1 (attack) or 1 (normal) to fit to Isolation Forest prediciton output
    #     y = [-1 if x != 'normal' else 1 for x in y]
    #     y_pred = [-1 if x != 'normal' else 1 for x in y_pred]

    #     print(len(y))
    #     print(len(y_pred))

    #     return f1_score(y, y_pred)
    def f1_scorer(estimator, X, y_true, **kwargs):
        y_true = [-1 if x != 'normal' else 1 for x in y_true]
        y_pred = estimator.predict(X)
        # convert to 0/1 labels
        y_pred = (y_pred == -1).astype(int)
        y_true = (y_true == -1).astype(int)
        return f1_score(y_true, y_pred)

    i_forest = IsolationForest(verbose=False, n_jobs=28)

    grid = {'n_estimators':[50, 100, 500],
            #'max_samples':[256],
            'max_samples':[50, 256, 1000],
            #'max_features':[0.1]}
            'max_features':[0.05, 0.1, 0.5]}
    gs = GridSearchCV(estimator=i_forest,
                      param_grid=grid,
                      scoring=make_scorer(f1_scorer),
                      cv=3,
                      return_train_score=True, verbose=False)
    gs.fit(fvs, labels)

    best_i_forest = gs.best_estimator_

    return best_i_forest, gs.best_params_

def train_test_rfc(features, nr_training_samples, nr_test_samples):

        # train on passed features
        train_fvs, train_labels = dp.get_fvs_from_parquet(parquet_paths=i.parquet_paths,
                                                        NR_ELEMENTS=nr_training_samples,
                                                        attack_types=i.attack_types,
                                                        all_samples=False)

        train_fvs = train_fvs[features]

        best_rfc, best_params_before, predictions, train_accuracy, feature_importances = rfc(train_fvs, train_labels)

        # test on passed features
        test_fvs, test_labels = dp.get_fvs_from_parquet(parquet_paths=i.parquet_paths,
                                                                NR_ELEMENTS=nr_test_samples,
                                                                attack_types=i.attack_types,
                                                                all_samples=False)
        test_fvs = test_fvs[features]

        test_predictions = best_rfc.predict(test_fvs)
        test_accuracy = accuracy_score(test_labels, test_predictions)

        return train_accuracy, test_accuracy
    
    
# region auxiliary -------------------------------------------------------------------------------------------------------------------------------

def save_to_pickle(data, path : dp.Path):
    dp.pp.save(path, data, overwrite=True)
    print("saved data")

def get_feature_names_from_csv(path : dp.Path):
    return dp.pd.read_csv(path).columns