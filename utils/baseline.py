# BASELINE LogisticRegression using all attributes with cross validation
from statistics import mean

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # FOR CATEGORICAL DATA
from sklearn.preprocessing import OrdinalEncoder


def prepare_input(X):
    oe = OrdinalEncoder()
    oe.fit(X)
    X_enc = oe.transform(X)
    return X_enc


# prepare target
def prepare_target(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    return y_enc


def prepare_all(X, y, test_size, random_state=1):
    X_enc = prepare_input(X)
    y_enc = prepare_target(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=test_size, random_state=1
    )
    return X_train, y_train, X_test


# FOR CROSS FOLD VALIDATION
from sklearn.model_selection import KFold, StratifiedKFold


def baseline_for_binary_with_all(X, y, number_of_folds):
    list_of_acc = []
    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)
    for train, test in skf.split(X, y):
        model = LogisticRegression(solver="lbfgs")
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        list_of_acc.append(accuracy_score(y[test], y_pred))
    return mean(list_of_acc)


def baseline_for_categorical_with_all(X, y, number_of_folds):
    list_of_acc = []
    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)
    X_enc = prepare_input(X)
    y_enc = prepare_target(y)
    for train, test in skf.split(X_enc, y_enc):
        model = LogisticRegression(solver="lbfgs")
        model.fit(X_enc[train], y_enc[train])
        y_pred = model.predict(X_enc[test])
        list_of_acc.append(accuracy_score(y_enc[test], y_pred))
    return mean(list_of_acc)


def accuracy(X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
