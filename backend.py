import openai
import re
import random
from IPython.display import clear_output
import re
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pickle
from sklearn.preprocessing import MinMaxScaler
import requests
import json
from catboost import CatBoostRegressor
base = pd.read_csv("base.csv")
A = base['price'].to_numpy()
base = base.drop("Unnamed: 0", axis=1)
B = base.drop('price', axis=1).to_numpy()
scaler = MinMaxScaler()
scaler2 = MinMaxScaler()
X_normalized = scaler.fit(B)
Y_norm = scaler2.fit(A.reshape(-1, 1))

model = CatBoostRegressor()
model.load_model("catboost_reg_model.bin")


def gptRequist(field):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer OPENAI API KEY"  
    }
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": field}],
        "temperature": 0.7
    }
    request_json = json.dumps(request_data)

    response = requests.post(url, headers=headers, data=request_json)
    response_data = response.json()
    generated_message = response_data["choices"][0]["message"]["content"]

    return generated_message

def getInfo(prompt):
    #adjustprompt = "Please provide the values in the following format:\n\nneed: value\ninfo: [key: value]\ninfo: [key, value]\n\nExample: need: 5, info: [name: John], info: [age: 25]. Here is the prompt --> " + prompt
    adjustprompt = "Lire attentivement cette demande (" + prompt + " et fournir les valeurs explicitement dans cette forme : Need : value : [key : value], info : [key, value]..., (Example : need : 5, info : [name : John], info : [age : 25])."
    text = gptRequist(adjustprompt)
    need_pattern = r'need:\s*([^,\n]+)'
    need_match = re.search(need_pattern, text)
    need_value = need_match.group(1).strip() if need_match else None

    info_pattern = r'\[([^:]+):\s*([^,\]]+)\]'
    matches = re.findall(info_pattern, text)
    key_value_pairs = []
    if need_value != None:
        key_value_pairs.append(('need', need_value))
    for match in matches:
        key = match[0].strip()
        value = match[1].strip()
        if key != 'need':  # Exclude 'need' from key_value_pairs
            key_value_pairs.append((key, value))
    
    return key_value_pairs


extract_integer = lambda s: int(''.join(filter(str.isdigit, s))) if any(char.isdigit() for char in s) else None
transform_yes_no = lambda s: 1 if s.lower() == "yes" or s.lower()=="true" else 0

def getFeat(listfeat):
    json_file_path = 'feat.json'

    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    if listfeat != None:
        list_of_feat = []
        for key, value in json_data.items():
            for feat in listfeat:
                if value in feat[0]:
                    if key not in ['waterfront']:
                        list_of_feat.append((key, extract_integer(feat[1])))
                    else:
                        list_of_feat.append((key, transform_yes_no(feat[1])))
        #list_of_feat.append(('need', dict(listfeat)['need']))
        list_of_feat.append(('need', 'price'))
    else:
        return None
    print(list_of_feat)
    return dict(list_of_feat)

def imputer(feat):
  df = pd.read_csv("data_1.csv")
  data = df.drop('id', axis=1)
  data['date'] = data['date'].str[4:6]
  col = list(data.columns)
  if feat==None:
      return None
  else:
    user_data = [0 for i in range(len(col))]
    for i in range(len(col)):
        user_data[i] = feat[col[i]] if col[i] in feat else np.nan

    X_missing = np.array(user_data)
    X_missing = np.reshape(X_missing, (1, X_missing.shape[0]))
    data = np.concatenate((data, X_missing), axis=0)

    knn_imputer = KNNImputer(n_neighbors=5)
    X_imputed = knn_imputer.fit_transform(data)

  return X_imputed[-1]

def catigories(df, column):
  one_hot_encoded = pd.get_dummies(df[column].astype(int), prefix=column)

  new_column_names = ['{}'.format(col) for col, value in zip(one_hot_encoded.columns, one_hot_encoded.columns)]
  one_hot_encoded.columns = [f'{x.split("_")[0]}_{int(float(x.split("_")[1]) * 10)}' for x in new_column_names]
  df = df.join(one_hot_encoded)

  return df.drop(column, axis=1)

def pre_process(imput):
    base = pd.read_csv("data_1.csv")
    columns = ['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    data_array = np.array(imput).reshape(1, -1)
    df = pd.DataFrame(data_array, columns=columns)
    df['floors'] = df['floors'].astype(int)
    df['view'] = df['view'].astype(int)
    df['condition'] = df['condition'].astype(int)
    df = catigories(df, 'floors')
    df = catigories(df, 'view')
    df = catigories(df, 'condition')
    base_year = 1900
    df['yr_built_encoded'] = df['yr_built'] - base_year
    df = df.drop('yr_built', axis=1)
    df['yr_renovated_encoded'] = df['yr_renovated'] - base_year
    df.loc[df['yr_renovated_encoded'] == -1900, 'yr_renovated_encoded'] = 0
    df = df.drop('yr_renovated', axis=1)
    zipcode_counts = base['zipcode'].value_counts()
    df['zipcode_freq_encoded'] = df['zipcode'].map(zipcode_counts)
    df = df.drop('zipcode', axis=1)
    df = catigories(df, 'date')
    featurs = ['Unnamed: 0', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'waterfront', 'grade', 'sqft_above', 'sqft_basement', 'lat',
       'long', 'sqft_living15', 'sqft_lot15', 'floors_10', 'floors_15',
       'floors_20', 'floors_25', 'floors_30', 'floors_35', 'view_0', 'view_10',
       'view_20', 'view_30', 'view_40', 'condition_10', 'condition_20',
       'condition_30', 'condition_40', 'condition_50', 'yr_built_encoded',
       'yr_renovated_encoded', 'zipcode_freq_encoded', 'date_10', 'date_20',
       'date_30', 'date_40', 'date_50', 'date_60', 'date_70', 'date_80',
       'date_90', 'date_100', 'date_110', 'date_120']
    output = pd.DataFrame()
    for item in featurs:
        if item not in df.columns:
            df[item] = 0
        output[item] = df[item]
    output.replace(True, 1, inplace=True)
    output = output.drop("Unnamed: 0", axis=1)
    output.to_csv("output.csv")
    return output

def predict(prompt):
    output = pre_process(list(imputer(getFeat(getInfo(prompt)))))
    X = output.drop('price', axis=1).to_numpy()
    X_normalized = scaler.transform(X)
    y = model.predict(X_normalized)
    return str(list(scaler2.inverse_transform(y.reshape(-1, 1)).reshape((1,)))[0])

def feedback(prompt):
    price = predict(prompt)
    adjustprompt = "write a short respense to this ask (" + prompt + ") by giving this suggation price: " + price
    return gptRequist(adjustprompt)


"""
prompt = "i want the price of a house in month 1 that contain 3 bedrooms and 1 bathroom and surface of living room is 250 with 1 floors in this zipcode 98178 that has water front with pool and basement of 2455"


print(feedback(prompt))"""




