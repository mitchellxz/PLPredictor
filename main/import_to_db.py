import psycopg2
from sqlalchemy import create_engine
import sys


def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1) 
    print("Connection successful")
    return conn

def to_alchemy(df):
    """
    Using a dummy table to test this call library
    """
    engine = create_engine(connect)
    df.to_sql(
        'main_match', 
        con=engine, 
        index=False, 
        if_exists='replace'
    )
    print("to_sql() done (sqlalchemy)")
    
    
#to_alchemy(preprocessed_df)