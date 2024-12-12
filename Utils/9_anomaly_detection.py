import numpy as np
import pandas as pd
from scipy.stats import norm

'''
training data: normal points only
validation and testing data: both normal and anomolous
'''

##### Anomaly detection #####
def get_epsilon(train_data, val_data, label_name):
    '''Get the threshold from training data'''
    val_data['overall_prob'] = 1

    for feature in train_data.drop(columns=[label_name]).columns:
        mean = train_data[feature].mean()
        std = train_data[feature].std()

        val_data[f'prob_{feature}'] = norm.pdf(val_data[feature], mean, std)
        val_data['overall_prob'] *= val_data[f'prob_{feature}']
        
    sorted_val = val_data.sort_values(by='overall_prob', ascending=False).reset_index(drop=True)
    index = sorted_val[sorted_val[label_name] == 1].index[0]
    epsilon = sorted_val.loc[index, 'overall_prob']

    return epsilon

def detect_anomalies(train_data, val_data, test_data, label_name):
    '''Classify anomalies in test_data'''
    test_data['overall_prob'] = 1

    for feature in test_data.drop(columns=[label_name]).columns:
        mean = train_data[feature].mean()
        std = train_data[feature].std()

        test_data[f'prob_{feature}'] = norm.pdf(val_data[feature], mean, std)
        test_data['overall_prob'] *= test_data[f'prob_{feature}']
    
    epsilon = get_epsilon(train_data, val_data, label_name)
    test_data['predicted_label'] = (test_data['overall_prob'] < epsilon).astype(int)

    return test_data