# Import important libraries

import pandas as pd
import numpy as np
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
class1_data = loadmat("D:/frequency_band_after_filt/Class1_freqband_10_all.mat")['class1_feat']
class2_data = loadmat("D:/frequency_band_after_filt/Class2_freqband_80_all.mat")['class2_feat']
class3_data = loadmat("D:/frequency_band_after_filt/Class3_freqband_0_all.mat")['Class3_feat']

# Initialize scaler
scaler = StandardScaler()

# Initialize results storage
acc = {
    'svm': [], 'random_forest': [], 'decision_tree': [],
    'gradient_boosting': [], 'extra_tree': [],
    'logistic_regression': [], 'naive_bayes': [], 'ridge_classifier': []
}

ind_band = []

# Initialize band data
class1 = class1_data[0, :, :]
class2 = class2_data[0, :, :]
class3 = class3_data[0, :, :]

for i in range(1, 9):
    print(f"Processing band {i}")
    
    # Combine and scale data
    result_combined = np.vstack((class1, class2, class3))
    scaler.fit(result_combined)
    result_combined = scaler.transform(result_combined)
    
    a, b, c = class1.shape[0], class2.shape[0], class3.shape[0]
    labels = np.array([0] * a + [1] * b + [2] * c)
    
    X_train, X_test, y_train, y_test = train_test_split(result_combined, labels, test_size=0.2, random_state=42)
    
    # Feature selection using Ridge
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    ind = np.abs(ridge.coef_) >= 0.005

    X_train_ridge = X_train[:, ind]
    X_test_ridge = X_test[:, ind]
    ind_band.append(ind)
    
    # Train and evaluate classifiers
    classifiers = {
        'svm': SVC(kernel='linear', C=100),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'decision_tree': DecisionTreeClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'extra_tree': ExtraTreeClassifier(random_state=42),
        'logistic_regression': LogisticRegression(),
        'naive_bayes': GaussianNB(),
        'ridge_classifier': ridge
    }

    for clf_name, clf in classifiers.items():
        clf.fit(X_train_ridge, y_train)
        y_pred = clf.predict(X_test_ridge)
        accuracy = accuracy_score(y_test, y_pred)
        acc[clf_name].append(accuracy)
        print(f"{clf_name} Accuracy: {accuracy:.2f}")

    # Update class data for the next iteration
    if i < 8:
        class1 = np.hstack((class1, class1_data[i, :, :]))
        class2 = np.hstack((class2, class2_data[i, :, :]))
        class3 = np.hstack((class3, class3_data[i, :, :]))

# Store results
results_dict = {'accuracies': acc, 'ind_bands': ind_band}

print("Final results:", results_dict)


