import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
#from load_model import get_model


#model = joblib.load(r'model/sc.pkl')
encoder = LabelEncoder()

def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def labelencoder(input_val,feats): 
    print("input :", input_val)
    print(" ")
    print("feats : ",feats)
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats    
    inpvalue = encoder.fit_transform(feat_val)
    inpdict = dict(zip(feat_key, inpvalue))
    value = inpdict[input_val]
    print("value",value)
    return value


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
    