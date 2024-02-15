import import_to_db
import match_data_preprocessor
import pandas as pd

def preprocess_data():
    ##preprocess data
    df = match_data_preprocessor.loadData()
    preprocessed_df, label, features = match_data_preprocessor.preprocess(df)
    preprocessed_df.insert(0, 'id', range(1, len(preprocessed_df) + 1))

    return preprocessed_df


def connect_to_server():
    ##Connect to db server
    param_dic = {
    "host"      : "localhost",
    "database"  : "PLData",
    "user"      : "postgres",
    "password"  : "05212002"
    }

    conn = import_to_db.connect(param_dic)

    import_to_db.connect = "postgresql+psycopg2://%s:%s@%s:5432/%s" % (
    param_dic['user'],
    param_dic['password'],
    param_dic['host'],
    param_dic['database']
    )

    import_to_db.to_alchemy(preprocessed_df)



preprocessed_df = pd.read_csv('mysite\main\static\main\csv\PLList.csv')

preprocessed_df = preprocess_data()
connect_to_server()
