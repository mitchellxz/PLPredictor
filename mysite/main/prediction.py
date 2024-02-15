import pandas as pd
def generate_features_for_match(home_team_name, away_team_name, columns):
    match_data = pd.DataFrame(columns=columns)
    match_data.loc[0] = 0

    home_team_name = preprocess_team_name(home_team_name)
    away_team_name = preprocess_team_name(away_team_name)

    match_data['HomeTeam_' + home_team_name] = 1
    match_data['AwayTeam_' + away_team_name] = 1

    print(match_data)

    return match_data

def results(home_team_name, away_team_name, features, model):
    Xnew = generate_features_for_match(home_team_name, away_team_name, features.columns)

    ynew = model.predict(Xnew)

    predicted_class = ynew[0]
    print(predicted_class)

    outcome_mapping = {
        0: away_team_name,
        1: "draw",
        2: home_team_name
    }

    expected_outcome = outcome_mapping.get(predicted_class)

    print("Home Team: ", home_team_name)
    print("Away Team: ", away_team_name)
    print("Predicted Outcome: ", expected_outcome)

    #self.predict_goals_based_on_outcome(home_team_name, away_team_name, predicted_class)

def preprocess_team_name(team_name):
        return team_name.replace("'m", "ingham")

def team_names(home_team, away_team):
     print(home_team)
     print(away_team)
     #features = pd.read_csv('mysite/main/static/main/csv/featureslist.csv')
   