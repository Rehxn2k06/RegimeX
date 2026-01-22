import pandas as pd 

def load_data(path):
    df=pd.read_csv(path,low_memory=False)

    return df