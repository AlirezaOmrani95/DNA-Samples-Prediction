'''
    This project is about prediction of the age of ancient DNA samples from their genomic data, and due to limited number of data, I had to use k-folding method.
'''

import pandas as pd
from keras.models import Sequential, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam

df = pd.read_excel('/media/aro/New Volume/Flash/motivation letter/Sweden/Lunds University/Doctoral student in Biology/Position/new data/new data.xlsx')


target_column = ['DateBP']

predictors = list(set(list(df.columns))-set(list(target_column)))
print(df[predictors].describe())

x = df[predictors].values

y = df[target_column].values

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.1,random_state = 40)

def model():
    model = Sequential([])
    model.add(Dense(32, kernel_initializer="normal", activation='relu', input_dim = 8))
    model.add(Dense(64, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1024, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(512, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(256, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, kernel_initializer="normal", activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, kernel_initializer="normal",  activation='relu'))
    model.add(Dense(16, kernel_initializer="normal",  activation='relu'))
    model.add(Dense(1, kernel_initializer="normal"))

    model.compile(loss='mse', optimizer=Adam(),metrics=['mae'])
    
    return model



m = model()

k = 7
num_val_samples = len(X_train)//k
num_epochs = 2000
all_scores = []

for i in range(k):
    print('fold {} of {}'.format(i+1,k))
    val_data = X_train[i*num_val_samples: (i+1)*num_val_samples]
    val_target = Y_train[i*num_val_samples: (i+1)*num_val_samples]
    train_data = np.concatenate([X_train[:i*num_val_samples],X_train[(i+1)*num_val_samples:]],axis=0)
    train_target = np.concatenate([Y_train[:i*num_val_samples],Y_train[(i+1)*num_val_samples:]],axis=0)
    history = m.fit(train_data,train_target,epochs=num_epochs)
    val_mse,val_mae = m.evaluate(val_data,val_target, verbose=1)
    all_scores.append(val_mae)
print(all_scores)
print(np.mean(all_scores))
print('---------------------------')
print('test data: ',m.evaluate(X_test,Y_test))

print('---------------------------')
a = m.predict(X_test)
for i in range(len(a)):
    print('True: {},   Predicted: {}'.format(Y_test[i],a[i]))