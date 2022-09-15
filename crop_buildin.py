import pandas as pd
crops = pd.read_csv('Crop_recommendation.csv')

df = crops.copy()
target = 'label'

target_mapper = {'rice':0, 'maize':1, 'chickpea':2, 'kidneybeans':3, 'pigeonpeas':4,
       'mothbeans':5, 'mungbean':6, 'blackgram':7, 'lentil':8, 'pomegranate':9,
       'banana':10, 'mango':11, 'grapes':12, 'watermelon':13, 'muskmelon':14, 'apple':15,
       'orange':16, 'papaya':17, 'coconut':18, 'cotton':19, 'jute':20, 'coffee':21}
def target_encode(val):
    return target_mapper[val]

df['label'] = df['label'].apply(target_encode)

Y = df['label']
X = df.drop('label',axis=1)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,Y)

import pickle
pickle.dump(clf, open('crops_clf.pkl','wb'))