from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd
import cupy as cp
from sklearn.naive_bayes import GaussianNB as sk_gnb
from cuml.naive_bayes import GaussianNB as cu_gnb
from cuml.ensemble import RandomForestClassifier as cu_rfc
from sklearn.ensemble import RandomForestClassifier as sk_rfc
import numpy as np
from sklearn.feature_selection import SelectFromModel
from cuml.linear_model import LogisticRegression as cu_lr
from imblearn.over_sampling import ADASYN
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score
from cuml.svm import LinearSVC as cu_linear_svc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import cudf
import joblib
import os
import cupy as cp
import gc
from sklearn.naive_bayes import GaussianNB
import os
import dataframe_image as dfi
from cuml.svm import SVC as cuSVC 
from sklearn.svm import LinearSVC as sk_linear_svc
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
cp.random.seed(RANDOM_SEED)

script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(script_path)
output_path = os.path.join(root_path, 'models')
input_path = os.path.join(root_path, 'data', 'processed')
os.makedirs(output_path, exist_ok=True)

class ClassificationMetricsCV:
    results_df = pd.DataFrame()

    def __init__(self, model, X, y):
        self.accuracy = []
        self.error_rate = []
        self.recall = []
        self.precision = []
        self.f1_score = []
        self.roc_auc = []
        self.model = model

        if isinstance(X, cp.ndarray):
            X = X.get()

        if isinstance(X, cudf.DataFrame):
            X = X.to_pandas()

        self.X = pd.DataFrame(X).reset_index(drop=True)

        if isinstance(y, cp.ndarray):
            y = y.get()

        if isinstance(y, cudf.Series):
            y = y.to_pandas()

        self.y = pd.Series(y).reset_index(drop=True)

    def compute_metrics(self, y_test, y_pred):
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        self.accuracy.append(acc)

        # Error rate (1 - accuracy)
        error_rate = 1 - acc
        self.error_rate.append(error_rate)

        # Recall
        recall = recall_score(y_test, y_pred, average='binary')
        self.recall.append(recall)

        # Precision
        precision = precision_score(y_test, y_pred, average='binary')
        self.precision.append(precision)

        # F1 score
        f1_score_result = f1_score(y_test, y_pred, average='binary')
        self.f1_score.append(f1_score_result)

        # ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred)
        self.roc_auc.append(roc_auc)

    def stratifiedKfold(self):
        skf = StratifiedKFold(n_splits=10, random_state=42,shuffle=True)
        for train_index, test_index in skf.split(self.X,self.y):

            X_train = self.X.iloc[train_index, :]
            X_test = self.X.iloc[test_index, :]
            y_train = self.y[train_index]
            y_test = self.y[test_index]

            self.model.fit(X_train.to_numpy(), y_train)
            y_pred = self.model.predict(X_test)

            self.compute_metrics(y_test,y_pred)

    def save_results(self,algorithm):
        results = {
            "Algorithm": algorithm,
            "accuracy": round(np.mean(self.accuracy), 4),
            "error_rate": round(np.mean(self.error_rate), 4),
            "recall": round(np.mean(self.recall), 4),
            "precision": round(np.mean(self.precision), 4),
            "f1_score": round(np.mean(self.f1_score), 4),
            "roc_auc": round(np.mean(self.roc_auc), 4),
        }
        ClassificationMetricsCV.results_df = pd.concat(
            [ClassificationMetricsCV.results_df, pd.DataFrame([results])],
            ignore_index=True
        )

    def printMetrics(self):
        print(f"accuracy: {round(np.mean(self.accuracy), 4)}")
        print(f"error_rate: {round(np.mean(self.error_rate), 4)}")
        print(f"recall: {round(np.mean(self.recall), 4)}")
        print(f"precision: {round(np.mean(self.precision), 4)}")
        print(f"f1_score: {round(np.mean(self.f1_score), 4)}")
        print(f"roc_auc: {round(np.mean(self.roc_auc), 4)}")

    def save_model(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(self.model, path)

    @staticmethod
    def printResults(save_as_image=True, image_path='results_table.png'):
        styled_df = (ClassificationMetricsCV.results_df.style
                    .hide(axis='index')  # Hide index for all outputs
                    .set_table_styles([
                        {'selector': 'th',
                        'props': [('background-color', '#f0f0f0'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'center'),
                                    ('color', 'black')]},
                        {'selector': 'td',
                        'props': [('text-align', 'center'),
                                    ('background-color', '#f0f0f0'),
                                    ('color', 'black')]},
                        {'selector': 'td:nth-child(1)',
                        'props': [('background-color', '#f0f0f0'),
                                    ('font-weight', 'bold')]}])
                    .format("{:.4f}", subset=ClassificationMetricsCV.results_df.columns.drop('Algorithm')))

        if save_as_image:
            try:
                import dataframe_image as dfi
                dfi.export(styled_df, image_path)
                print(f"Results table saved as image: {image_path}")
            except Exception as e:
                print(f"Warning: Could not save image due to error: {str(e)}")
                print("Saving as HTML instead...")
                try:
                    # Fallback: Save as HTML (index already hidden by styled_df)
                    html_path = image_path.replace('.png', '.html')
                    with open(html_path, 'w') as f:
                        f.write(styled_df._repr_html_())
                    print(f"Results table saved as HTML: {html_path}")
                except Exception as html_error:
                    print(f"Could not save as HTML either: {str(html_error)}")
                    # Final fallback: just display the DataFrame
                    print("Displaying results instead:")
                    print(ClassificationMetricsCV.results_df.to_string(index=False))
        else:
            try:
                # For notebook environments
                from IPython.display import display
                display(styled_df)  # Index already hidden in styled_df
            except ImportError:
                # Fallback for non-notebook environments
                print(ClassificationMetricsCV.results_df.to_string(index=False))


df = pd.read_parquet(os.path.join(input_path, 'selected_features_df.parquet'))


X = df.drop(columns=['TARGET'])
y = df['TARGET']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y) 
x_resampled = X_resampled.astype(np.float32)


#######
# LOGISTIC REGRESSION
#######
logisticRegression = cu_lr()
selector = SelectFromModel(estimator=logisticRegression).fit(X_resampled, y_resampled)
X_selected = selector.transform(X_resampled)

def objective(trial):
        cp._default_memory_pool.free_all_blocks()
        C = trial.suggest_float('C', 1e-4, 1, log = True)
        max_iter = trial.suggest_int('max_iter', 100, 10000)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        
        logisticRegression = cu_lr(C=C, max_iter=max_iter, class_weight=class_weight)    
        scoring = {'roc_auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}
        cv_results = cross_validate(
            logisticRegression,
            X_selected,
            y_resampled,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )

        mean_roc_auc = cv_results['test_roc_auc'].mean()
        mean_precision = cv_results['test_precision'].mean()
        mean_recall = cv_results['test_recall'].mean()

        metric = np.mean([mean_roc_auc, mean_precision, mean_recall])

        del logisticRegression, cv_results
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        
        return metric

output_model_path = os.path.join(output_path, 'logistic_regression')
os.makedirs(output_model_path, exist_ok=True)

study = optuna.create_study(direction='maximize', study_name="Logistic Regression")
study.optimize(objective, n_trials=30, gc_after_trial=True)

model = cu_lr(**study.best_params)
lr_metrics = ClassificationMetricsCV(model,X_selected,y_resampled)
lr_metrics.save_model(os.path.join(output_model_path, 'logistic_regression.pkl'))
lr_metrics.stratifiedKfold()
lr_metrics.save_results('Logistic Regression')
lr_metrics.printResults(save_as_image=True, image_path=os.path.join(output_model_path, 'logistic_regression_results.png'))
# Save Optuna Study Plots
optuna.visualization.plot_optimization_history(study).write_image(os.path.join(output_model_path, 'logistic_regression_optimization_history.png'))
optuna.visualization.plot_slice(study).write_image(os.path.join(output_model_path, 'logistic_regression_slice.png'))
optuna.visualization.plot_contour(study).write_image(os.path.join(output_model_path, 'logistic_regression_contour.png'))
optuna.visualization.plot_parallel_coordinate(study).write_image(os.path.join(output_model_path, 'logistic_regression_parallel_coordinate.png'))
optuna.visualization.plot_param_importances(study).write_image(os.path.join(output_model_path, 'logistic_regression_param_importances.png'))
optuna.visualization.plot_edf(study).write_image(os.path.join(output_model_path, 'logistic_regression_edf.png'))


#######
# RANDOM FOREST
#######
print("Selecting Random Forest")
randomForest = sk_rfc()
selector = SelectFromModel(estimator=randomForest).fit(X_resampled, y_resampled)
X_selected = selector.transform(X_resampled)
X_selected = X_selected.astype(np.float32)
print("Random Forest selected")

def objective(trial):
    gc.collect()
    cp._default_memory_pool.free_all_blocks()
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    
    random_forest_classifier = cu_rfc(**param)
    
    scoring = {'roc_auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}
    
    cv_results = cross_validate(
        random_forest_classifier,
        X_selected,
        y_resampled,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_precision = cv_results['test_precision'].mean()
    mean_recall = cv_results['test_recall'].mean()
    
    metric = np.mean([mean_roc_auc, mean_precision, mean_recall])
    
    # del random_forest_classifier, cv_results
    del cv_results
    gc.collect()
    cp._default_memory_pool.free_all_blocks()

    return metric

output_model_path = os.path.join(output_path, 'random_forest')
os.makedirs(output_model_path, exist_ok=True)

study = optuna.create_study(direction='maximize', study_name="Random Forest")
study.optimize(objective, n_trials=20, gc_after_trial=True)

model = cu_rfc(**study.best_params)
randomForest_metrics = ClassificationMetricsCV(model,X_selected,y_resampled)
randomForest_metrics.save_model(os.path.join(output_model_path, 'random_forest.pkl'))
randomForest_metrics.stratifiedKfold()
randomForest_metrics.save_results('Random Forest')
randomForest_metrics.printResults(save_as_image=True, image_path=os.path.join(output_model_path, 'random_forest_results.png'))
# Save Optuna Study Plots
optuna.visualization.plot_optimization_history(study).write_image(os.path.join(output_model_path, 'random_forest_optimization_history.png'))
optuna.visualization.plot_slice(study).write_image(os.path.join(output_model_path, 'random_forest_slice.png'))
optuna.visualization.plot_contour(study).write_image(os.path.join(output_model_path, 'random_forest_contour.png'))
optuna.visualization.plot_parallel_coordinate(study).write_image(os.path.join(output_model_path, 'random_forest_parallel_coordinate.png'))
optuna.visualization.plot_param_importances(study).write_image(os.path.join(output_model_path, 'random_forest_param_importances.png'))
optuna.visualization.plot_edf(study).write_image(os.path.join(output_model_path, 'random_forest_edf.png'))

#######
# NAIVE BAYES
#######

def objective(trial):

        prior1 = trial.suggest_float('prior1', 0.0, 1.0)
        prior2 = 1.0 - prior1
        priors = cp.array([prior1, prior2])

        gnb_classifier = cu_gnb(priors=priors)

        scoring = {'roc_auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}
        cv_results = cross_validate(
            gnb_classifier,
            X_resampled,
            y_resampled,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )

        mean_roc_auc = cv_results['test_roc_auc'].mean()
        mean_precision = cv_results['test_precision'].mean()
        mean_recall = cv_results['test_recall'].mean()

        metric = np.mean([mean_roc_auc, mean_precision, mean_recall])
        
        del gnb_classifier, cv_results
        gc.collect()
        cp._default_memory_pool.free_all_blocks()


        return metric

output_model_path = os.path.join(output_path, 'naive_bayes')
os.makedirs(output_model_path, exist_ok=True)

study = optuna.create_study(direction='maximize',study_name="Naive Bayes")
study.optimize(objective, n_trials=50, gc_after_trial=True)

model = cu_gnb(**study.best_params)
naiveBayes_metrics = ClassificationMetricsCV(model,X_resampled,y_resampled)
naiveBayes_metrics.save_model(os.path.join(output_model_path, 'naive_bayes.pkl'))
naiveBayes_metrics.stratifiedKfold()
naiveBayes_metrics.save_results('Naive Bayes')
naiveBayes_metrics.printResults(save_as_image=True, image_path=os.path.join(output_model_path, 'naive_bayes_results.png'))
# Save Optuna Study Plots
optuna.visualization.plot_optimization_history(study).write_image(os.path.join(output_model_path, 'naive_bayes_optimization_history.png'))
optuna.visualization.plot_slice(study).write_image(os.path.join(output_model_path, 'naive_bayes_slice.png'))
optuna.visualization.plot_contour(study).write_image(os.path.join(output_model_path, 'naive_bayes_contour.png'))
optuna.visualization.plot_parallel_coordinate(study).write_image(os.path.join(output_model_path, 'naive_bayes_parallel_coordinate.png'))
optuna.visualization.plot_param_importances(study).write_image(os.path.join(output_model_path, 'naive_bayes_param_importances.png'))
optuna.visualization.plot_edf(study).write_image(os.path.join(output_model_path, 'naive_bayes_edf.png'))

#######
# SVM
#######

svm = sk_linear_svc()
selector = SelectFromModel(estimator=svm).fit(X_resampled, y_resampled)
X_selected = selector.transform(X_resampled)
X_selected = X_selected.astype(np.float32)

def objective(trial):
    gc.collect()
    cp._default_memory_pool.free_all_blocks()
    C = trial.suggest_float('C', 1e-5, 1e2,log=True)
    tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
    
    svm_classifier = cu_linear_svc(C=C, tol=tol, max_iter=max_iter, loss=loss)
    
    scoring = {'roc_auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}
    cv_results = cross_validate(
        svm_classifier,
        X_selected,
        y_resampled,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )

    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_precision = cv_results['test_precision'].mean()
    mean_recall = cv_results['test_recall'].mean()

    metric = np.mean([mean_roc_auc, mean_precision, mean_recall])
    
    del svm_classifier, cv_results
    gc.collect()
    cp._default_memory_pool.free_all_blocks()
    
    return metric

output_model_path = os.path.join(output_path, 'linear_svm')
os.makedirs(output_model_path, exist_ok=True)

study = optuna.create_study(direction='maximize', study_name="LinearSVM")
study.optimize(objective, n_trials=20, gc_after_trial=True)

model = cu_linear_svc(**study.best_params)
svm_metrics = ClassificationMetricsCV(model,X_selected,y_resampled)
svm_metrics.save_model(os.path.join(output_model_path, 'linear_svm.pkl'))
svm_metrics.stratifiedKfold()
svm_metrics.save_results('LinearSVM')
svm_metrics.printResults(save_as_image=True, image_path=os.path.join(output_model_path, 'linear_svm_results.png'))
# Save Optuna Study Plots
optuna.visualization.plot_optimization_history(study).write_image(os.path.join(output_model_path, 'linear_svm_optimization_history.png'))
optuna.visualization.plot_slice(study).write_image(os.path.join(output_model_path, 'linear_svm_slice.png'))
optuna.visualization.plot_contour(study).write_image(os.path.join(output_model_path, 'linear_svm_contour.png'))
optuna.visualization.plot_parallel_coordinate(study).write_image(os.path.join(output_model_path, 'linear_svm_parallel_coordinate.png'))
optuna.visualization.plot_param_importances(study).write_image(os.path.join(output_model_path, 'linear_svm_param_importances.png'))
optuna.visualization.plot_edf(study).write_image(os.path.join(output_model_path, 'linear_svm_edf.png'))

# #######
# # NON-LINEAR SVM
# #######

svm = sk_linear_svc()
selector = SelectFromModel(estimator=svm).fit(X_resampled, y_resampled)
X_selected = selector.transform(X_resampled)
X_selected = X_selected.astype(np.float32)

def objective(trial):
    gc.collect()
    cp._default_memory_pool.free_all_blocks()
    C = trial.suggest_float('C', 1e-5, 1e2,log=True)
    gamma = trial.suggest_float('gamma', 1e-5, 1e2,log=True)
    tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    
    svm_classifier = cuSVC(C=C, tol=tol, max_iter=max_iter, kernel=kernel, gamma = gamma)
    
    scoring = {'roc_auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}
    cv_results = cross_validate(
        svm_classifier,
        X_selected,
        y_resampled,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_precision = cv_results['test_precision'].mean()
    mean_recall = cv_results['test_recall'].mean()

    metric = np.mean([mean_roc_auc, mean_precision, mean_recall])

    del svm_classifier, cv_results
    gc.collect()
    cp._default_memory_pool.free_all_blocks()

    return metric

output_model_path = os.path.join(output_path, 'non_linear_svm')
os.makedirs(output_model_path, exist_ok=True)

study = optuna.create_study(direction='maximize', study_name="Non-LinearSVM")
study.optimize(objective, n_trials=20, gc_after_trial=True)

model = cuSVC(**study.best_params)
svm_metrics = ClassificationMetricsCV(model,X_selected,y_resampled)
svm_metrics.save_model(os.path.join(output_model_path, 'non_linear_svm.pkl'))
svm_metrics.stratifiedKfold()
svm_metrics.save_results('Non-LinearSVM')
svm_metrics.printResults(save_as_image=True, image_path=os.path.join(output_model_path, 'non_linear_svm_results.png'))
# Save Optuna Study Plots
optuna.visualization.plot_optimization_history(study).write_image(os.path.join(output_model_path, 'non_linear_svm_optimization_history.png'))
optuna.visualization.plot_slice(study).write_image(os.path.join(output_model_path, 'non_linear_svm_slice.png'))
optuna.visualization.plot_contour(study).write_image(os.path.join(output_model_path, 'non_linear_svm_contour.png'))
optuna.visualization.plot_parallel_coordinate(study).write_image(os.path.join(output_model_path, 'non_linear_svm_parallel_coordinate.png'))
optuna.visualization.plot_param_importances(study).write_image(os.path.join(output_model_path, 'non_linear_svm_param_importances.png'))
optuna.visualization.plot_edf(study).write_image(os.path.join(output_model_path, 'non_linear_svm_edf.png'))

#######
# CART
#######

cart = DecisionTreeClassifier()
selector = SelectFromModel(estimator=cart).fit(X_resampled, y_resampled)
X_selected = selector.transform(X_resampled)
X_selected = X_selected.astype(np.float32)

def objective(trial):
    gc.collect()
    cp._default_memory_pool.free_all_blocks()
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy','log_loss'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])

    cart = DecisionTreeClassifier(criterion = criterion, min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, class_weight=class_weight, random_state=42)

    scoring = {'roc_auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}
    cv_results = cross_validate(
        cart,
        X_selected,
        y_resampled,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_precision = cv_results['test_precision'].mean()
    mean_recall = cv_results['test_recall'].mean()

    metric = np.mean([mean_roc_auc, mean_precision, mean_recall])
    
    del cart, cv_results
    gc.collect()
    cp._default_memory_pool.free_all_blocks()
    
    return metric

output_model_path = os.path.join(output_path, 'cart')
os.makedirs(output_model_path, exist_ok=True)

study = optuna.create_study(direction='maximize', study_name="CART")
study.optimize(objective, n_trials=20, gc_after_trial=True)

model = DecisionTreeClassifier(**study.best_params)
cart_metrics = ClassificationMetricsCV(model,X_selected,y_resampled)
cart_metrics.save_model(os.path.join(output_model_path, 'cart.pkl'))
cart_metrics.stratifiedKfold()
cart_metrics.save_results('CART')
cart_metrics.printResults(save_as_image=True, image_path=os.path.join(output_model_path, 'cart_results.png'))
# Save Optuna Study Plots
optuna.visualization.plot_optimization_history(study).write_image(os.path.join(output_model_path, 'cart_optimization_history.png'))
optuna.visualization.plot_slice(study).write_image(os.path.join(output_model_path, 'cart_slice.png'))
optuna.visualization.plot_contour(study).write_image(os.path.join(output_model_path, 'cart_contour.png'))
optuna.visualization.plot_parallel_coordinate(study).write_image(os.path.join(output_model_path, 'cart_parallel_coordinate.png'))
optuna.visualization.plot_param_importances(study).write_image(os.path.join(output_model_path, 'cart_param_importances.png'))
optuna.visualization.plot_edf(study).write_image(os.path.join(output_model_path, 'cart_edf.png'))


#######
# Print Results
#######

print(f"Logistic Regression: {logistic_regression_metrics.results_df}")
print(f"Random Forest: {randomForest_metrics.results_df}")
print(f"Naive Bayes: {naiveBayes_metrics.results_df}")
print(f"LinearSVM: {svm_metrics.results_df}")
print(f"Non-LinearSVM: {svm_metrics.results_df}")
print(f"CART: {cart_metrics.results_df}")

ClassificationMetricsCV.printResults(save_as_image=True, image_path=os.path.join(output_path, 'results_table.png'))