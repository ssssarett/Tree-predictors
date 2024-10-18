import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed

class TreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, threshold=None, left=None, right=None, is_categorical=False):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_categorical = is_categorical

    def predict(self, x):
        if self.is_leaf:
            return self.prediction
        else:
            value = x[self.feature_index]
            if self.is_categorical:
                if value == self.threshold:
                    return self.left.predict(x)
                else:
                    return self.right.predict(x)
            else:
                if value <= self.threshold:
                    return self.left.predict(x)
                else:
                    return self.right.predict(x)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, impurity_measure='gini', max_features=None, feature_names=None, categorical_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_measure = impurity_measure
        self.max_features = max_features
        self.feature_names = feature_names
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.root = None
        self.n_features = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self.root.predict(sample) for sample in X])

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (num_samples < self.min_samples_split) or \
           (num_labels == 1):
            leaf_value = self._majority_vote(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)

        if self.max_features is None:
            feature_indices = np.arange(self.n_features)
        else:
            feature_indices = np.random.choice(self.n_features, self.max_features, replace=False)

        # best split
        best_feature, best_threshold, best_impurity, impurity_reduction, best_is_categorical = self._best_split(X, y, feature_indices)

        if best_feature is None:
            leaf_value = self._majority_vote(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)

        # update feature importance
        self.feature_importances_[best_feature] += impurity_reduction

        # split dataset
        if best_is_categorical:
            left_indices = X[:, best_feature] == best_threshold
            right_indices = X[:, best_feature] != best_threshold
        else:
            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold

        # build children recursively
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(
            is_leaf=False,
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            is_categorical=best_is_categorical
        )

    def _best_split(self, X, y, feature_indices):
        num_samples = len(y)
        best_impurity = float('inf')
        best_feature = None
        best_threshold = None
        best_impurity_reduction = 0
        best_is_categorical = False

        parent_impurity = self._node_impurity(y)

        for feature_index in feature_indices:
            X_column = X[:, feature_index]

            if feature_index in self.categorical_features:
                # categorical feature
                categories = np.unique(X_column)
                for category in categories:
                    left_mask = X_column == category
                    right_mask = X_column != category

                    y_left = y[left_mask]
                    y_right = y[right_mask]

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue

                    impurity = self._calculate_impurity_split(y_left, y_right)

                    if impurity < best_impurity:
                        best_impurity = impurity
                        best_feature = feature_index
                        best_threshold = category  
                        best_impurity_reduction = parent_impurity - impurity
                        best_is_categorical = True
            else:
                # numerical feature
                sorted_indices = X_column.argsort()
                X_sorted = X_column[sorted_indices]
                y_sorted = y[sorted_indices]


                unique_values = np.unique(X_sorted)
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

                for threshold in thresholds:
                    left_mask = X_sorted <= threshold
                    right_mask = X_sorted > threshold

                    y_left = y_sorted[left_mask]
                    y_right = y_sorted[right_mask]

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue

                    impurity = self._calculate_impurity_split(y_left, y_right)

                    if impurity < best_impurity:
                        best_impurity = impurity
                        best_feature = feature_index
                        best_threshold = threshold
                        best_impurity_reduction = parent_impurity - impurity
                        best_is_categorical = False

        return best_feature, best_threshold, best_impurity, best_impurity_reduction, best_is_categorical

    def _calculate_impurity_split(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        n_left = len(y_left)
        n_right = len(y_right)

        impurity_left = self._node_impurity(y_left)

        impurity_right = self._node_impurity(y_right)

        impurity_total = (n_left / n) * impurity_left + (n_right / n) * impurity_right

        return impurity_total

    def _node_impurity(self, y):
        counts = np.bincount(y)
        probabilities = counts / counts.sum()

        if self.impurity_measure == 'gini':
            impurity = 1 - np.sum(probabilities ** 2)
        elif self.impurity_measure == 'entropy':
            impurity = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        elif self.impurity_measure == 'sqrt':
            impurity = np.sum(np.sqrt(probabilities * (1 - probabilities)))
        else:
            raise ValueError("Invalid impurity measure. Choose 'gini', 'entropy', or 'sqrt'.")

        return impurity

    def _majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return values[index]

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, impurity_measure='gini', max_features=None, feature_names=None, categorical_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_measure = impurity_measure
        self.max_features = max_features
        self.feature_names = feature_names
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.trees = []
        self.feature_importances_ = None
        self.n_features = None

    def fit(self, X, y):
        self.trees = []
        n_samples, self.n_features = X.shape

        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(self.n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(self.n_features))
            else:
                self.max_features = None

        self.trees = Parallel(n_jobs=-1)(
            delayed(self._train_tree)(X, y)
            for _ in range(self.n_estimators)
        )

        self.compute_feature_importances()

    def _train_tree(self, X, y):
        n_samples = X.shape[0]
        # bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            impurity_measure=self.impurity_measure,
            max_features=self.max_features,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features
        )
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        y_pred = []
        for sample_preds in tree_preds.T:
            values, counts = np.unique(sample_preds, return_counts=True)
            index = np.argmax(counts)
            y_pred.append(values[index])
        return np.array(y_pred)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def compute_feature_importances(self):
        if not self.trees:
            raise ValueError("The Random Forest has not been trained.")

        total_importances = np.zeros(self.n_features)

        for tree in self.trees:
            tree_importances = tree.feature_importances_
            if tree_importances is not None:
                total_importances += tree_importances

        self.feature_importances_ = total_importances / np.sum(total_importances)
        return self.feature_importances_

######## DATA PREPROCESSING ########
mushroom = pd.read_csv('MushroomDataset/secondary_data.csv', sep=';')

### Exploratory analysis
print(f"Number of observations: {mushroom.shape[0]}")
print(f"Number of variables: {mushroom.shape[1]}")

#### 1) Missing values
## Columns
missing_values = mushroom.isnull().sum()
print("Missing values per column:")
print(missing_values)

missing_percentage = mushroom.isnull().mean() * 100
columns_to_drop = missing_percentage[missing_percentage > 25].index
data = mushroom.drop(columns=columns_to_drop)

print(f"Number of observations after dropping columns with >25% missing values: {data.shape[0]}")
print(f"Number of variables after dropping columns with >25% missing values: {data.shape[1]}")

## Rows
data = data.dropna()

print(f"Number of observations after dropping rows with missing values: {data.shape[0]}")
print(f"Number of variables: {data.shape[1]}")

#### 2) Duplicates
data = data.drop_duplicates()
print(f"Number of observations after removing duplicates: {data.shape[0]}")

######## ENCODING CATEGORICAL VARIABLES ########

def encode_features(df):
    df_encoded = pd.DataFrame()
    encoders = {}
    categorical_features = []
    for i, column in enumerate(df.columns):
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            df_encoded[column], uniques = pd.factorize(df[column])
            encoders[column] = uniques
            categorical_features.append(i)
        else:
            df_encoded[column] = df[column]
    return df_encoded, encoders, categorical_features

X = data.drop('class', axis=1)
y = data['class']

X_encoded, X_encoders, categorical_features = encode_features(X)
y_encoded, y_encoders = pd.factorize(y)

X_encoded = X_encoded.values
y_encoded = y_encoded

######## SPLITTING DATASET BEFORE TUNING ########
np.random.seed(42)
indices = np.arange(X_encoded.shape[0])
np.random.shuffle(indices)
X_encoded = X_encoded[indices]
y_encoded = y_encoded[indices]

# 80% training, 20% test
split_index = int(0.8 * X_encoded.shape[0])
X_train_full, X_test = X_encoded[:split_index], X_encoded[split_index:]
y_train_full, y_test = y_encoded[:split_index], y_encoded[split_index:]

######## HYPERPARAMETER TUNING ########

# Cross-validation K-fold
def k_fold_cross_validation(X, y, k, model_params):
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    accuracy_list = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size if fold != k - 1 else n_samples

        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        tree = DecisionTree(**model_params, categorical_features=categorical_features)
        tree.fit(X_train, y_train)

        accuracy = tree.evaluate(X_val, y_val)
        accuracy_list.append(accuracy)

    return accuracy_list

def evaluate_params(args):
    impurity, max_depth, min_samples_split = args
    model_params = {
        'impurity_measure': impurity,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }

    accuracies = k_fold_cross_validation(X_train_full, y_train_full, k=5, model_params=model_params)
    mean_accuracy = np.mean(accuracies)

    print(f"Tested: impurity={impurity}, max_depth={max_depth}, "
          f"min_samples_split={min_samples_split}, mean accuracy={mean_accuracy * 100:.2f}%")

    return {
        'impurity_measure': impurity,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'mean_accuracy': mean_accuracy
    }

impurity_measures = ['gini', 'entropy', 'sqrt']  # Added 'sqrt'
max_depths = [3, 5, 9, None]
min_samples_splits = [2, 5, 10]

### TUNING
k = 5

param_grid = list(itertools.product(impurity_measures, max_depths, min_samples_splits))

results = Parallel(n_jobs=-1)(
    delayed(evaluate_params)(params)
    for params in param_grid
)

results_df = pd.DataFrame(results)

best_result = results_df.loc[results_df['mean_accuracy'].idxmax()]

print("\nBest hyperparameters found:")
print(f"Impurity Measure: {best_result['impurity_measure']}")
print(f"Max Depth: {best_result['max_depth']}")
print(f"Min Samples Split: {best_result['min_samples_split']}")
print(f"Mean Accuracy: {best_result['mean_accuracy'] * 100:.2f}%")

final_tree = DecisionTree(
    impurity_measure=best_result['impurity_measure'],
    max_depth=best_result['max_depth'],
    min_samples_split=int(best_result['min_samples_split']),
    categorical_features=categorical_features
)
final_tree.fit(X_train_full, y_train_full)

test_accuracy = final_tree.evaluate(X_test, y_test)
print(f"Test accuracy with best hyperparameters: {test_accuracy * 100:.2f}%")

train_accuracy = final_tree.evaluate(X_train_full, y_train_full)
print(f"Training accuracy with best hyperparameters: {train_accuracy * 100:.2f}%")

y_pred_train = final_tree.predict(X_train_full)
training_error = np.mean(y_pred_train != y_train_full)
print(f"Training error (0-1 loss) with best hyperparameters: {training_error * 100:.2f}%")

def compute_confusion_matrix(y_true, y_pred, positive_label=1):
    TP = np.sum((y_true == positive_label) & (y_pred == positive_label))
    FP = np.sum((y_true != positive_label) & (y_pred == positive_label))
    TN = np.sum((y_true != positive_label) & (y_pred != positive_label))
    FN = np.sum((y_true == positive_label) & (y_pred != positive_label))
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

def compute_performance_metrics(confusion):
    TP = confusion['TP']
    FP = confusion['FP']
    TN = confusion['TN']
    FN = confusion['FN']

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    }

y_pred_test = final_tree.predict(X_test)
y_pred_train = final_tree.predict(X_train_full)

## CONFUSION MATRIX
confusion_test = compute_confusion_matrix(y_test, y_pred_test, positive_label=1)
confusion_train = compute_confusion_matrix(y_train_full, y_pred_train, positive_label=1)

print("Confusion Matrix Test:")
print(f"\tPredicted Negative\tPredicted Positive")
print(f"Actual Negative\t{confusion_test['TN']}\t\t\t{confusion_test['FP']}")
print(f"Actual Positive\t{confusion_test['FN']}\t\t\t{confusion_test['TP']}")

print("\nConfusion Matrix Training:")
print(f"\tPredicted Negative\tPredicted Positive")
print(f"Actual Negative\t{confusion_train['TN']}\t\t\t{confusion_train['FP']}")
print(f"Actual Positive\t{confusion_train['FN']}\t\t\t{confusion_train['TP']}")

## PERFORMANCE METRICS
metrics_test = compute_performance_metrics(confusion_test)
metrics_train = compute_performance_metrics(confusion_train)

print("\nTest Set Performance Metrics:")
print(f"Accuracy: {metrics_test['Accuracy'] * 100:.2f}%")
print(f"Precision: {metrics_test['Precision'] * 100:.2f}%")
print(f"Recall: {metrics_test['Recall'] * 100:.2f}%")
print(f"F1-Score: {metrics_test['F1-Score'] * 100:.2f}%\n")

print("Training Set Performance Metrics:")
print(f"Accuracy: {metrics_train['Accuracy'] * 100:.2f}%")
print(f"Precision: {metrics_train['Precision'] * 100:.2f}%")
print(f"Recall: {metrics_train['Recall'] * 100:.2f}%")
print(f"F1-Score: {metrics_train['F1-Score'] * 100:.2f}%")

######## RANDOM FOREST ########

rf = RandomForest(
    n_estimators=10,
    max_depth=None,
    min_samples_split=2,
    impurity_measure=best_result['impurity_measure'],  # Use the best impurity measure
    max_features='sqrt',
    categorical_features=categorical_features
)

rf.fit(X_train_full, y_train_full)


test_accuracy_rf = rf.evaluate(X_test, y_test)
print(f"\nRandom Forest test accuracy: {test_accuracy_rf * 100:.2f}%")

train_accuracy_rf = rf.evaluate(X_train_full, y_train_full)
print(f"Random Forest training accuracy: {train_accuracy_rf * 100:.2f}%")


# Predictions
y_pred_test_rf = rf.predict(X_test)
y_pred_train_rf = rf.predict(X_train_full)

### CONFUSION MATRIX
confusion_test_rf = compute_confusion_matrix(y_test, y_pred_test_rf, positive_label=1)
confusion_train_rf = compute_confusion_matrix(y_train_full, y_pred_train_rf, positive_label=1)

print("Confusion Matrix Test (Random Forest):")
print(f"\tPredicted Negative\tPredicted Positive")
print(f"Actual Negative\t{confusion_test_rf['TN']}\t\t\t{confusion_test_rf['FP']}")
print(f"Actual Positive\t{confusion_test_rf['FN']}\t\t\t{confusion_test_rf['TP']}")

print("\nConfusion Matrix Training (Random Forest):")
print(f"\tPredicted Negative\tPredicted Positive")
print(f"Actual Negative\t{confusion_train_rf['TN']}\t\t\t{confusion_train_rf['FP']}")
print(f"Actual Positive\t{confusion_train_rf['FN']}\t\t\t{confusion_train_rf['TP']}")


### PERFORMANCE METRICS 
metrics_test_rf = compute_performance_metrics(confusion_test_rf)
metrics_train_rf = compute_performance_metrics(confusion_train_rf)

print("\nRandom Forest Test Set Performance Metrics:")
print(f"Accuracy: {metrics_test_rf['Accuracy'] * 100:.2f}%")
print(f"Precision: {metrics_test_rf['Precision'] * 100:.2f}%")
print(f"Recall: {metrics_test_rf['Recall'] * 100:.2f}%")
print(f"F1-Score: {metrics_test_rf['F1-Score'] * 100:.2f}%\n")

print("Random Forest Training Set Performance Metrics:")
print(f"Accuracy: {metrics_train_rf['Accuracy'] * 100:.2f}%")
print(f"Precision: {metrics_train_rf['Precision'] * 100:.2f}%")
print(f"Recall: {metrics_train_rf['Recall'] * 100:.2f}%")
print(f"F1-Score: {metrics_train_rf['F1-Score'] * 100:.2f}%")

