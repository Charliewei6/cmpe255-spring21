import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        self.X_test = None
        self.y_test = None
        

    def define_feature(self,feature_selection):
        if feature_selection==0:
            feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        elif feature_selection==1:
            feature_cols = ['pregnant', 'insulin', 'glucose','bmi']
        elif feature_selection==2:
            feature_cols = ['pregnant', 'pedigree', 'skin', 'glucose', 'bmi']
        elif feature_selection==3:
            feature_cols = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'pedigree']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train(self,feature_selection):
        # split X and y into training and testing sets
        X, y = self.define_feature(feature_selection)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self,feature_selection):
        model = self.train(feature_selection)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    print("Experiement      Accuracy              Confusion Matrix            Comment")
    classifer = DiabetesClassifier()
    result = classifer.predict(0)
    score = classifer.calculate_accuracy(result)
    con_matrix = classifer.confusion_matrix(result)
    print("Baseline       "f"{score}""      "f"[{con_matrix[0]} {con_matrix[1]}]")
    # print(f"confusion_matrix=${con_matrix}")

    classifer = DiabetesClassifier()
    result = classifer.predict(1)
    score = classifer.calculate_accuracy(result)
    con_matrix = classifer.confusion_matrix(result)
    print("Solution 1     "f"{score}""      "f"[{con_matrix[0]} {con_matrix[1]}]""    pregnant,insulin,glucose,bmi")
    
    classifer = DiabetesClassifier()
    result = classifer.predict(2)
    score = classifer.calculate_accuracy(result)
    con_matrix = classifer.confusion_matrix(result)
    print("Solution 2     "f"{score}""                "f"[{con_matrix[0]} {con_matrix[1]}]" "    pregnant,pedigree,skin,glucose,bmi")

    classifer = DiabetesClassifier()
    result = classifer.predict(3)
    score = classifer.calculate_accuracy(result)
    con_matrix = classifer.confusion_matrix(result)
    print("Solution 3     "f"{score}""      "f"[{con_matrix[0]} {con_matrix[1]}]""    pregnant,glucose,bp,insulin,bmi,pedigree")
    
