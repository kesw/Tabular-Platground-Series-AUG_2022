import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib

def preprocess(dataframe):
    ### turn 'attribute_0' and 'attribute_1' into an int
    dataframe['attribute_0'] = dataframe['attribute_0'].str[-1]
    dataframe['attribute_1'] = dataframe['attribute_1'].str[-1]
    dataframe['attribute_0'] = dataframe['attribute_0'].apply(pd.to_numeric)
    dataframe['attribute_1'] = dataframe['attribute_1'].apply(pd.to_numeric)

    ### use LabelEncode from sklearn for 'product_code' labeling
    labelencoder = LabelEncoder()
    data_le=pd.DataFrame(dataframe['product_code'])
    dataframe['product_code'] = labelencoder.fit_transform(data_le['product_code'])


    null_col = dataframe.columns[dataframe.isna().any()].tolist()
    for col in null_col:
        dataframe[col] = dataframe[col].interpolate(method="linear", limit_direction="both")
    
    #dataframe = dataframe.fillna(dataframe.mean())

    return dataframe



model = joblib.load('model')
df_test=pd.read_csv("test.csv")
df_test = preprocess(df_test)
x_val = df_test.drop("id", axis=1)

y_pred = model.predict_proba(x_val)

sub = pd.DataFrame({'id': df_test.index + 26570, 'failure': y_pred[:,1]})
sub.to_csv('109550193_submission.csv', index=False)
