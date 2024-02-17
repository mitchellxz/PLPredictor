import pandas as pd

def generate_features_for_match(home_team_name, away_team_name, columns):
    match_data = pd.DataFrame(columns=columns)
    match_data.loc[0] = 0

    match_data['HomeTeam_' + home_team_name] = 1
    match_data['AwayTeam_' + away_team_name] = 1

    return match_data

def getResults(home_team_name, away_team_name, features, model):

    home_team_name = get_rid_of_space(home_team_name)
    away_team_name = get_rid_of_space(away_team_name)

    Xnew = generate_features_for_match(home_team_name, away_team_name, features.columns)

    ynew = model.predict(Xnew)

    predicted_class = ynew[0]
    print(predicted_class)

    outcome_mapping = {
        0: away_team_name,
        1: "Draw",
        2: home_team_name
    }

    expected_outcome = outcome_mapping.get(predicted_class)
    expected_outcome = add_space(expected_outcome)

    print("Home Team: ", home_team_name)
    print("Away Team: ", away_team_name)
    print("Predicted Outcome: ", expected_outcome)

    return expected_outcome

    #self.predict_goals_based_on_outcome(home_team_name, away_team_name, predicted_class)

def get_rid_of_space(team_name):
        return team_name.replace(" ", "_")

def add_space(team_name):
     return team_name.replace("_", " ")