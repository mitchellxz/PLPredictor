import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def trainingTesting(label, features):
    y = np.ravel(label)
    X = features
    cv = StratifiedKFold(n_splits=10)

    model = MultinomialNB()

    accuracies = [] # list to store accuracies for each fold

    #self.predictScores(X)

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        #Make predictions on test set
        predictions = model.predict(X_test)

        #Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        
    overall_accuracy = np.mean(accuracies)
    print(f"Overall Accuracy: {overall_accuracy}")

    return model