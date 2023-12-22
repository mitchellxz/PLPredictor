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

        self.df = self.loadData()
        self.label, self.features = self.preprocess(self.df)
        self.model = self.trainingTesting(self.df, self.label, self.features)
        
        self.setWindowTitle("FTGH vs FTR")
        self.plot_fthg_vs_ftr()
    
  
    def plot_fthg_vs_ftr(self):
        ftr_categories = self.df['FTR'].unique()
        num_categories = len(ftr_categories)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        for i, feature in enumerate(['AS', 'HS']):
            ax = axs[i]
            ax.set_title(f'{feature} by Full Time Result (FTR)')
            ax.set_xlabel(f'{feature}')
            ax.set_ylabel('Frequency')

            for ftr in ftr_categories:
                ax.hist(self.df[self.df['FTR'] == ftr][feature], bins=15, alpha=0.5, label=f'FTR: {ftr}')

            ax.legend()
            ax.grid(axis='y', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def action(self):
        home_team_name = self.combobox1.currentText()
        away_team_name = self.combobox2.currentText()

        self.results(home_team_name, away_team_name, self.features, self.model)

    def loadData(self):
        csv_url = 'https://www.football-data.co.uk/mmz4281/2324/E0.csv'
        df = pd.read_csv(csv_url)
        
        return df


    def preprocess(self, df):
        columns_to_keep = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS']
        df = df[columns_to_keep].copy()

        #Replace 'm in Nott'm Forest
        df['HomeTeam'] = df['HomeTeam'].str.replace("'m", "ingham")
        df['AwayTeam'] = df['AwayTeam'].str.replace("'m", "ingham")

        df.to_csv(r'CSVFiles\PLList_pre.csv', encoding='utf-8-sig', index = False)

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
        features = df.iloc[:,0:]

        df.to_csv(r'CSVFiles\PLList.csv', encoding='utf-8-sig', index = False)

        features.to_csv(r'CSVFiles/featureslist.csv', encoding='utf-8-sig', index=False)
        
        label.to_csv(r'CSVFiles/labellist.csv', encoding='utf-8-sig', index=False)

        return label, features
    


    def trainingTesting(self, df, label, features):
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold
        import numpy as np

        y = np.ravel(label)
        X = features
        cv = StratifiedKFold(n_splits=10)

        model = MultinomialNB()

        accuracies = []  # List to store accuracies for each fold

        self.predictScores(X)

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)

            # Make predictions on the test set
            predictions = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)

        overall_accuracy = np.mean(accuracies)
        print(f"Overall Accuracy: {overall_accuracy}")

        return model

    #wip
    def predictScores(self, X):
        average_fthg = X.iloc[:, 0].mean()
        print(f"Average of FTHG column: {average_fthg}")

        average_ftag = X.iloc[:, 1].mean()
        print(f"Average of FTAG column: {average_ftag}")

        totalavg = (average_ftag + average_fthg)
        print(f"total average: {totalavg}") 

        print (average_fthg / totalavg)
        print (average_ftag / totalavg) 
        
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
    

    def results(self, home_team_name, away_team_name, features, model):

        Xnew = self.generate_features_for_match(home_team_name, away_team_name, features.columns)
        

        ynew = model.predict(Xnew)

        predicted_class = ynew[0]
        
        outcome_mapping = {
            0: away_team_name,
            1: "draw",
            2: home_team_name
        }
        
        expected_outcome = outcome_mapping.get(predicted_class)
        
        
        print("Home Team: ", home_team_name)
        print("Away Team: ", away_team_name)
        print("Predicted Outcome: ", expected_outcome)
    

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
