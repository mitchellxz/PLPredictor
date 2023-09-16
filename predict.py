import sys
from PySide6.QtWidgets import *
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Predictor")
        self.setGeometry(100, 100, 400, 200)
        
        self.combobox1 = QComboBox()
        
        csv_url = 'https://www.football-data.co.uk/mmz4281/2324/E0.csv'
        df = pd.read_csv(csv_url)
        
        column_names = df.columns
        
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
        text1 = self.combobox1.currentText()
        text2 = self.combobox2.currentText()
        print("Combo Box 1:", text1)
        print("Combo Box 2:", text2)
        
        # Call another method and pass text1 and text2 as arguments
        self.results_from_button(text1, text2)
        self.loadData()

    def results_from_button(self, text1, text2):
        # Perform some action with text1 and text2
        result = f"Received text1: {text1}, text2: {text2}"
        print(result)
        # You can do more with the result here, such as displaying it in a QMessageBox or updating the UI.
    
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

        df.to_csv(r'C:\Users\school\Documents\practice api python\PLList.csv', encoding='utf-8-sig', index = False)


        print(df.isnull().values.sum())


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
        print('result for match in row 1 is ', label[2])
        features = df.iloc[:,3:]

        df.to_csv(r'C:\Users\school\Documents\practice api python\PLList.csv', encoding='utf-8-sig', index = False)
        
        self.trainingTesting(df, label, features)
        
    def trainingTesting(self, df, label, features):
        y=np.ravel(label)

        X = features
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= .1, shuffle = False)

        print("X_train is " + str(X_train.shape))
        print("y_train is " + str(y_train.shape))
        print("X_test is " + str(X_test.shape))
        print("y_test is " + str(y_test.shape))


        y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

        print("size of y_train is " + str(y_train.shape))
        print("size of y_test is " + str(y_test.shape))
        print(y_train[0])


        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(117, input_dim=39, activation='relu'),
            tf.keras.layers.Dense(10, input_dim=117, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.summary()
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=17)

        plt.plot(history.history['accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.plot(history.history['loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        
        self.results(features, model)
        
        
    def preprocess_team_name(self, team_name):
        return team_name.replace("'m", "ingham")


    def generate_features_for_match(self, home_team_name, away_team_name, columns):
        # Create a DataFrame with the specified columns
        match_data = pd.DataFrame(columns=columns)

        # Initialize all columns to 0
        match_data.loc[0] = 0

        # Preprocess team names
        home_team_name = self.preprocess_team_name(home_team_name)
        away_team_name = self.preprocess_team_name(away_team_name)

        # Set the values for home and away teams based on your preprocessing
        match_data['HomeTeam_' + home_team_name] = 1
        match_data['AwayTeam_' + away_team_name] = 1

        return match_data
    
    def results(self, features, model):
        home_team_name = self.combobox1.currentText()
        away_team_name = self.combobox2.currentText()

        columns = features.columns  # Specify the columns based on your training data
        Xnew = self.generate_features_for_match(home_team_name, away_team_name, columns)

        # Convert Xnew to a comma-separated string
        Xnew_str = ', '.join(map(str, Xnew.iloc[0]))

        print("X = %s " % Xnew)
        ynew = np.argmax(model.predict(Xnew), axis=-1)

        print("Home Team: ", home_team_name)
        print("Away Team: ", away_team_name)
        print("Predicted Outcome: ", ynew[0])
        


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()