import json

import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def apply_classification_algorithms(train_file, test_file):
    # Load data from CSV files
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Extract features and labels from the datasets
    X_train = train_data.drop(
            columns=['source_id', 'target_id', 'pair_id', 'label']
            ).values
    y_train = train_data['label']
    X_test = test_data.drop(
            columns=['source_id', 'target_id', 'pair_id', 'label']
            ).values
    y_test = test_data['label']
    # Initialize classifiers
    classifiers = {
        # "Random Forest": RandomForestClassifier(),
        # "Logistic Regression": LogisticRegression(),
        "Support Vector Machine linear": SVC(kernel='linear'),
        "Support Vector Machine poly": SVC(kernel='poly'),
        "Support Vector Machine rbf": SVC(kernel='rbf'),
        "Support Vector Machine sigmoid": SVC(kernel='sigmoid'),
        # "K-Nearest Neighbors": KNeighborsClassifier()
    }

    # Dictionary to store evaluation metrics
    evaluation_metrics = {}
    # Apply each classifier and calculate accuracy on the test set
    for algo_name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='binary')
        precision = precision_score(y_test, predictions, average='binary')
        recall = recall_score(y_test, predictions, average='binary')
        evaluation_metrics[algo_name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        }
    return evaluation_metrics

# Example usage
train_file = 'features_train.csv'
test_file = 'features_test.csv'
accuracy_dict = apply_classification_algorithms(train_file, test_file)
print(json.dumps(accuracy_dict, indent=2))
