import pickle
import pandas as pd

with open('Task_4_data.pkl', 'rb') as f:
    data: pd.DataFrame = pickle.load(f)

    print(type(data))
    print(data)
    print(data.columns)
