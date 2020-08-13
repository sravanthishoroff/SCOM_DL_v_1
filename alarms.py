
import tensorflow_hub as hub

import pandas as pd
import numpy as np
import tensorflow as tf
import os, re, json , pickle
from keras import utils
import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from string import punctuation
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# import re

from sklearn import preprocessing

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report

def scom_alarm(dataframe):
    results = []
    df=pd.read_excel(dataframe)
    df.head()
    input_alarm_count = {"title":"Total count",
                            'count': df.shape[0]}
    results.append(input_alarm_count)

    # dropping NA in Ignorable column
    df['Ignorable (Y/N)'].dropna(how='all')
    # converting Y/N to 1 and 0 
    df['Ignorable (Y/N)']=df['Ignorable (Y/N)'].apply(lambda x: 1 if (x)=="Y" else 0)
    # df['Ignorable (Y/N)']
    # Dropping NA values in Alerts column
    df['Alerts'].dropna(how='all')
    # applying lower function to Alerts coulmn
    df['Alerts'] = df['Alerts'].apply(lambda x: x.lower())
    # df['Alerts']
    # applying regex for Alerts column
    df['Alerts'] = df['Alerts'].apply(lambda s: re.sub(r"[^a-zA-Z0-9]"," ",s))
    # return df['Alerts']

    lemmatizer=WordNetLemmatizer()

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        tokens =[w for w in tokens if not w in stop_words] # [w for w in
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
        return lemmatized_output

    # applying tokenize function to Alerts column
    df['Alerts'] = df['Alerts'].apply(tokenize)

    # encoding Alerts column 
    # alert_encoding = open(r"encoder.pickle","rb")
    # alert_dict = pickle.load(alert_encoding) 
    le = preprocessing.LabelEncoder()
    df['Alerts'] = le.fit_transform(df['Alerts'])
    # df['Alerts'] = alert_dict.transform(df['Alerts'])


    #taking X and y
    X = df['Alerts']
    y = df['Ignorable (Y/N)']

    #Scaling
    scaler = StandardScaler()

    #meta train test split
    X_train_meta, X_test_meta, y_train, y_test = train_test_split((X), (y) ,stratify=y, test_size=0.20,random_state=10)

    #scale X
    standardized = scaler.fit(X_train_meta.values.reshape(-1,1))
    X_train_std = standardized.transform(X_train_meta.values.reshape(-1,1))
    X_test_std = standardized.transform(X_test_meta.values.reshape(-1,1))

    # embedding 
    embedding_128 = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"


    #creating the embedding confuguration 
    hub_layer = hub.KerasLayer(embedding_128, input_shape=[], 
                            dtype=tf.string, trainable=True)
    
    # preprocessing for Support Queue column i.e., 2nd X variable
    df['Support Queue'] =  df['Support Queue'].replace(to_replace='Wintel server support',value='Wintel Server Support')
    df['Support Queue'] =  df['Support Queue'].replace(to_replace='wintel server Application',value='Wintel Server Applications')
    df['Support Queue'] =  df['Support Queue'].replace(to_replace = 'citrix',value='Citrix')
    df['Support Queue'] =  df['Support Queue'].replace(to_replace = ['database , backup','Backup , Wintel server support','Wintel server support , citrix','Citrix Server Support','depends on the server name','Wintel Server Support , database','CL - Service Account Lockout','Database , Backup','backup'],value='Others')
    df['Support Queue'] =  df['Support Queue'].replace(to_replace = 'Backup ',value='Backup Support')

    X_supportQueue = df[['Support Queue']]

    X_train_nlp, X_test_nlp, y_train, y_test = train_test_split(X_supportQueue, (y) ,stratify=y, test_size=0.20,random_state=10)

    ###Train Data
    embed = (hub_layer(X_train_nlp.values.reshape(-1,)))
    embed_df = pd.DataFrame((embed.numpy()[:].reshape(-1,128)))
    train_std_df = pd.DataFrame(X_train_std)
    concat_df = pd.concat([train_std_df,embed_df] ,axis=1)

    ##Test Data
    embed_test = (hub_layer(X_test_nlp.values.reshape(-1,)))
    embed_test_df = pd.DataFrame((embed_test.numpy()[:].reshape(-1,128)))
    test_std_df = pd.DataFrame(X_test_std)
    concat_test_df = pd.concat([test_std_df,embed_test_df] ,axis=1)

    # building layers
    input2 = tf.keras.layers.Input(shape=(129,))
    dense_layer_1 = tf.keras.layers.Dense(32, activation='relu')(input2)
    drop1= tf.keras.layers.Dropout(.2)
    dense_layer_2 = tf.keras.layers.Dense(32, activation='relu')(dense_layer_1)
    drop1= tf.keras.layers.Dropout(.2)
    dense_layer_3 = tf.keras.layers.Dense(32, activation='relu')(dense_layer_2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer_3)
    model = tf.keras.Model(inputs=input2, outputs=output)

    # building a model
    # model.summary() 

    # model.compile(optimizer='Adam',
    #               #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

    # loading model
    model = tf.keras.models.load_model("SCOM_sample.h5")

    y_pred = model.predict_on_batch(concat_df.values)

    output = pd.DataFrame(columns=['y_pred'],data = y_pred)
    output['pred_val'] = output['y_pred'].apply(lambda x:1 if (x) >= .58 else 0)

    #Generate false alarm excel
    # results = []
    false_output = output[output['pred_val']==0]
    # print(false_output.shape)
    false_alarm_count = {"title":"False Alarm",
                            'count' : false_output.shape[0]}
    results.append(false_alarm_count)

    output_df_false = df[df.index.isin(false_output.index)]
    output_df_false.to_excel('false_alarm_report.xls')

    #Generating true alarms from predicted output
    true_output = output[output['pred_val']==1]
    # print(true_output.shape)
    true_alarm_count = {'title':"True Alarm ",
                            'count':true_output.shape[0]}
    results.append(true_alarm_count)

    output_df_true =df[df.index.isin(true_output.index)]
    # print(output_df_true)
    # print(output_df_true.shape)

    return results
  
  


