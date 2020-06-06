# Titanic Survivability Project - Find out how many passengers survived
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Output results
def outputClassifierPredictions(predictor, classifierType=None):
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictor})
    output.to_csv('Titanic_Survival_Results_' + classifierType + '_classifier.csv', index=False)

# Show decision tree results
def decisionTreeGraph(data):
    export_graphviz(
        tree_clf,
        out_file=data,
        feature_names=data.columns,
        class_names=data["Survived"],
        rounded=True,
        filled=True
    )

if __name__ == '__main__':

    # Load training and test data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    # Peep into training data
    print(train_data.head(10))

    # Look at info of training data
    print(train_data.info())

    # Perform simple descriptive stats
    print(train_data.describe())

    # Define X (features) and y (target)
    features = ["Pclass", "Name", "Sex", "SibSp", "Parch"]
    X_train = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(train_data[features])
    y_train = train_data["Survived"]

    # Look at first 10 elements of X_train
    X_train.head(10)

    # Train data with a binary classifier model (start with SGD classifier)
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)

    # Apply model to test data
    predictor_SGD = sgd_clf.predict(X_test)

    # Apply predictor on model and save results
    outputClassifierPredictions(predictor_SGD, 'SGD')

    # Compute accuracy score of SGD classifier using cross-validation
    cross_val_score(sgd_clf, X_train, y_train, cv=5, scoring='accuracy')

    # Apply RandomForest Classifier
    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(X_train, y_train)

    # Apply model to test data
    predictor_forest = forest_clf.predict(X_test)

    # Apply predictor on model and save results
    outputClassifierPredictions(forest_clf, 'RandomForest')

    # Apply DecisionTree Classifier
    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X_train, y_train)

    # Apply model to test data
    predictor_tree = tree_clf.predict(X_test)

    # Apply predictor on model and save results
    outputClassifierPredictions(tree_clf, 'DecisionTree')



