import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Cancer:
    def __init__(self):
        self._filePath = None
        self._fileName = None
        self._df = None
        self._targetVariable = None

    def setDataFrame(self, df):
        self._df = df

    def getDataFrame(self):
        return self._df

    def readFile(self, fileName):
        self._fileName = fileName
        self._df=pd.read_csv(self._fileName)
        
    def getDfInfo(self):
        return self._df.info()
        
    def getFileHead20(self):
        return self._df.head(20)
        
    def getValueCounts(self):
        return self._df['class'].value_counts()
        
    def dropNaValues(self):
        self._df.dropna(inplace=True)
        
    def dropColumnFromDataFrame(self, columnName):
        self._df.drop ([columnName], axis=1)

    def setTargetVariable(self, columnName):
        self._targetVariable = self._df[columnName]
        
    def getTargetVariable(self):
        return self._targetVariable

    
    
    def randomForestModel(self):
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        #from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(self._df, self._targetVariable, test_size=0.4, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train_scaled, y_train)

        # Vorhersagen treffen
        y_pred = classifier.predict(X_test_scaled)

        # Confusion Matrix und Klassifizierungsbericht ausgeben
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def svmModel(self):
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = train_test_split(self._df, self._targetVariable, test_size=0.4, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svm_classifier = SVC(random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

        # Vorhersagen treffen
        y_pred_resampled = svm_classifier.predict(X_test_scaled)

        print(confusion_matrix(y_test, y_pred_resampled))
        print(classification_report(y_test, y_pred_resampled))


    
   

#a = Cancer()
#a.readFile("Lastversion_preprocess_ML.xlsx")

