import prediction
import train_test_model
import pandas as pd

def main(home_team, away_team):

    features = pd.read_csv('mysite/main/static/main/csv/featureslist.csv')
    label = pd.read_csv('mysite\main\static\main\csv\labellist.csv')
    model = train_test_model.trainingTesting(label, features)
    ##make prediction
    #prediction.results(home_team, away_team, features, model)
