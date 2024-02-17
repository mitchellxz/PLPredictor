import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def loadData():
    csv_url = 'https://www.football-data.co.uk/mmz4281/2324/E0.csv'
    df = pd.read_csv(csv_url)

    return df

def preprocess(df):
    
    columns_to_keep = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'Referee']
    df = df[columns_to_keep].copy()

    #Replace 'm in Nott'm Forest
    df['HomeTeam'] = df['HomeTeam'].str.replace("'m", "ingham")
    df['AwayTeam'] = df['AwayTeam'].str.replace("'m", "ingham")

    teams_with_spaces = ['Crystal Palace', 'Aston Villa', 'Man City', 'Man United', 'Nottingham Forest', 'Sheffield United', 'West Ham']
    for team in teams_with_spaces:
        df['HomeTeam'] = df['HomeTeam'].str.replace(team, team.replace(' ', '_'))
        df['AwayTeam'] = df['AwayTeam'].str.replace(team, team.replace(' ', '_'))

    df.to_csv(r'mysite\main\static\main\csv\PLList_pre.csv', encoding='utf-8-sig', index = False)

    df = pd.get_dummies(df, columns=['HomeTeam'], prefix=['HomeTeam'])
    df = pd.get_dummies(df, columns=['AwayTeam'], prefix=['AwayTeam'])

    df['Referee'] = preprocessReferee(df)
    df['FTR'] = preprocessFTR(df)

    label = df['FTR']
    features = df.iloc[:,0:]

    df.to_csv(r'mysite\main\static\main\csv\PLList.csv', encoding='utf-8-sig', index = False)
    features.to_csv(r'mysite\main\static\main\csv\featureslist.csv', encoding='utf-8-sig', index=False)
    label.to_csv(r'mysite\main\static\main\csv\labellist.csv', encoding='utf-8-sig', index=False)

    return df, label, features

def preprocessReferee(df):
    label_encoder = LabelEncoder()
    df['Referee'] = label_encoder.fit_transform(df['Referee'])

    referee_mode_mapping = {}

    # Creates a new DataFrame or series named 'grouped', where each row corresponds to a unique referee,
    # and the 'FTR' value is the most common outcome ('H', 'A', or 'D').
    grouped = df.groupby('Referee')['FTR'].agg(lambda x: Counter(x).most_common(1)[0][0])

    # Creating a mapping of referees to their most frequent outcome in 'FTR' and a corresponding mapped value
    for referee, mode_ftr in grouped.items():
        referee_mode_mapping[referee] = {
            'mode_FTR': mode_ftr,
            'mapped_value': 2 if mode_ftr == 'H' else (0 if mode_ftr == 'A' else 1)
        }
        
    # transform all values in 'Referee' column to the 'mapped_value' from referee_mode_mapping
    df['Referee'] = df['Referee'].map(lambda x: referee_mode_mapping[x]['mapped_value'])

    return df['Referee']


#transform categorical labels ('H', 'A', 'D') into numerical values
def preprocessFTR(df):
    label_encoder = LabelEncoder()
    df['FTR'] = label_encoder.fit_transform(df['FTR'])

    print(df['FTR'].unique())

    print('Home Win:  ', df['FTR'][3])
    print('Away Win: ', df['FTR'][0])
    print('Draw: ', df['FTR'][2])

    return df['FTR']


#df = loadData()
#preprocessed_df, label, features = preprocess(df)