import numpy as np 
import pandas as pd 
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
import joblib
warnings.filterwarnings('ignore')

df=pd.read_csv("train.csv")

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

df = preprocess(df)
x = df.drop("id", axis=1)
x = x.drop("failure", axis=1)
y = df["failure"]

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=42)

x_re, y_re = SMOTE(random_state=42).fit_resample(x_train, y_train)

svm=MLPClassifier(hidden_layer_sizes=(100, 100, 100), batch_size=64, learning_rate_init=0.001, early_stopping=True, verbose=True, solver="sgd", learning_rate="adaptive", random_state=42)
svm.fit(x_re, y_re)

joblib.dump(svm, 'model')