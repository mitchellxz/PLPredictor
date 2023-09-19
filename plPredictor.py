import sys
from PySide6.QtWidgets import *
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np
from numpy import mean
from numpy import sqrt
from numpy import absolute



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Predictor")
        self.setGeometry(100, 100, 400, 200)
        
        self.combobox1 = QComboBox()
        
        csv_url = 'https://www.football-data.co.uk/mmz4281/2324/E0.csv'
        df = pd.read_csv(csv_url)
                
        hometeam_names = df['HomeTeam'].unique()
        awayteam_names = df['AwayTeam'].unique()
        
        hometeam_list = list(hometeam_names)
        awayteam_list = list(awayteam_names)
        
        self.combobox1.addItems(hometeam_list)
        
        self.combobox2 = QComboBox()
        self.combobox2.addItems(awayteam_list)
                
        layout = QVBoxLayout()
        layout.addWidget(self.combobox1)
        layout.addWidget(self.combobox2)
        
        container = QWidget()
        container.setLayout(layout)
        
        self.setCentralWidget(container)
        
        button = QPushButton("Confirm", self)
        layout.addWidget(button)
        
        button.pressed.connect(self.action)
        
    
    def action(self):
        self.combobox1.currentText()
        self.combobox2.currentText()
        
        self.loadData()

    def loadData(self):
        csv_url = 'https://www.football-data.co.uk/mmz4281/2324/E0.csv'
        df = pd.read_csv(csv_url)
        
        self.preprocess(df)
                
    def preprocess(self, df):
        columns_to_keep = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        df = df[columns_to_keep].copy()

        #Replace 'm in Nott'm Forest
        df['HomeTeam'] = df['HomeTeam'].str.replace("'m", "ingham")
        df['AwayTeam'] = df['AwayTeam'].str.replace("'m", "ingham")

        df.to_csv(r'C:\Users\Mitchell (School)\Desktop\PythonCSVFiles\PLList.csv', encoding='utf-8-sig', index = False)

        df = pd.get_dummies(df, columns=['HomeTeam'], prefix=['HomeTeam'])
        df = pd.get_dummies(df, columns=['AwayTeam'], prefix=['AwayTeam'])


        label_encoder = LabelEncoder()
        df['FTR'] = label_encoder.fit_transform(df['FTR'])

        print('Unique values for our label are: ',
            df['FTR'].unique())

        print('if home team wins label is ', df['FTR'][3])
        print('if away team wins label is ', df['FTR'][0])
        print('if draw the label is ', df['FTR'][2])

        label = df['FTR']
        features = df.iloc[:,3:]

        df.to_csv(r'C:\Users\Mitchell (School)\Desktop\PythonCSVFiles\PLList.csv', encoding='utf-8-sig', index = False)
        
        self.trainingTesting(df, label, features)
        
    def trainingTesting(self, df, label, features):
        y=np.ravel(label)
        X = features
        
        cv = LeaveOneOut()

        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X, y)


        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        print(sqrt(mean(absolute(scores))))

        predicted = cross_val_predict(model, X, y, cv=cv)

        conf_matrix = confusion_matrix(y, predicted)

        precision_scores = []
        for i in range(len(conf_matrix)):
            true_positives = conf_matrix[i, i]
            false_positives = sum(conf_matrix[j, i] for j in range(len(conf_matrix)) if j != i)
            precision = true_positives / (true_positives + false_positives)
            precision_scores.append(precision)
        
        self.results(features, model)
        
        
    def preprocess_team_name(self, team_name):
        return team_name.replace("'m", "ingham")


    def generate_features_for_match(self, home_team_name, away_team_name, columns):
        match_data = pd.DataFrame(columns=columns)

        match_data.loc[0] = 0

        home_team_name = self.preprocess_team_name(home_team_name)
        away_team_name = self.preprocess_team_name(away_team_name)

        match_data['HomeTeam_' + home_team_name] = 1
        match_data['AwayTeam_' + away_team_name] = 1

        return match_data
    
    def results(self, features, model):
        home_team_name = self.combobox1.currentText()
        away_team_name = self.combobox2.currentText()

        columns = features.columns
        Xnew = self.generate_features_for_match(home_team_name, away_team_name, columns)

        ynew = model.predict(Xnew)

        predicted_class = ynew[0]

        print("Home Team: ", home_team_name)
        print("Away Team: ", away_team_name)
        print("Predicted Outcome: ", predicted_class)
        


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()