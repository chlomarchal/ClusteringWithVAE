# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:58:43 2023

@author: chloe
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections     import Counter
from sklearn.metrics import log_loss

data = pd.read_csv("C:/Users/chloe/OneDrive/Documents/DATS2M_1/MEMOIRE/Code/code officiel/data/train.csv", sep=',')
data2 = pd.read_csv("C:/Users/chloe/OneDrive/Documents/DATS2M_1/MEMOIRE/Code/code officiel/data/test.csv", sep=',')
data = pd.concat([data, data2])
data['Arrival Delay in Minutes'].fillna(value=data['Arrival Delay in Minutes'].median(axis=0),inplace=True)
data = data.drop(columns = ['Unnamed: 0', 'id']) 

# we convert the data into categories
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location','Food and drink','Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'satisfaction']:
    data[col]=data[col].astype('category')


# we create age categories
labels = ["7 - 18", "19 - 30", "31 - 40", "41 - 50", "51 - 64", "65-85"]
tmp = pd.cut(data['Age'], [7, 19, 31, 41, 51, 65, 85], right=False, labels=labels)
data = data.assign(AgeCat=tmp)

# we create FlightDistanceCat
labels = ["0 - 1000","1001 - 2000","2001 - 3000","3001 - 4000", "4001 - 5000"]
tmp  = pd.cut(data['Flight Distance'], [0,1000,2000,3000,4000, 5000], right=False ,labels=labels)
data = data.assign(FlightDistanceCat=tmp)

# we create DepartDelayCat and ArrivalDelayCat
labels = ["0 - 5", "6 - 60","61 - 120","121 - 240","240+"]
tmp  = pd.cut(data['Departure Delay in Minutes'], [0,5,60,120,240,1600], right=False ,labels=labels)
data = data.assign(DepartDelayCat=tmp)
tmp  = pd.cut(data['Arrival Delay in Minutes'], [0,5,60,120,240, 1600], right=False ,labels=labels)
data = data.assign(ArrivalDelayCat=tmp)

# we create a dummy for satisfaction
data['satisfaction'] = data['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})


Y        =  data[['satisfaction']]
X        =  data.drop(['Age','Flight Distance','Departure Delay in Minutes',
                       'Arrival Delay in Minutes', 'satisfaction'],axis=1)

#%%
D  =  pd.get_dummies(X,drop_first=False)
l  =  X.shape[1]  
B  =  np.dot(np.asarray(D,dtype=int).T , np.asarray(D,dtype=int))   # Burt matrix or D.T @ D
C  =  np.diag(1/np.sqrt(np.diag(B))) 
Bw =  1/l * C @ B @ C
X_ind = np.asarray(D) @ Bw /l

original_indices = np.arange(len(X_ind))

# Split the transformed dataset into train and test sets
train_indices, test_indices = train_test_split(original_indices, test_size=0.2, random_state=2023)

# Retrieve samples for train and test sets based on indices
X_train_ind = X_ind[train_indices]
X_test_ind = X_ind[test_indices]

X_train = X.iloc[train_indices, :]
X_test = X.iloc[test_indices, :]

y_train = Y.iloc[train_indices, :]
y_test = Y.iloc[test_indices, :]

#%% Function to compute the clusters and logloss in function of the number of clusters clus

def burt(clus) : 
    km = KMeans(init="random",
                    n_clusters=clus,
                    n_init=20,
                    max_iter=300,
                    random_state=42)  # seed of the rnd number generator
    km.fit(X_train_ind)
        
    X_clus = km.predict(X_test_ind)
            # vector of clusters associated to each record
            #X_clus = km.labels_
            # out_tab : tables of dominant profiles in each cluster
    out_names = ['Gender', 'Customer Type', 'Type of Travel', 'Class',
                         'Inflight wifi service', 'Departure/Arrival time convenient',
                         'Ease of Online booking', 'Gate location', 'Food and drink',
                         'Online boarding', 'Seat comfort', 'Inflight entertainment',
                         'On-board service', 'Leg room service', 'Baggage handling',
                         'Checkin service', 'Inflight service', 'Cleanliness', 'AgeCat',
                         'FlightDistanceCat', 'DepartDelayCat', 'ArrivalDelayCat', 'satisfaction']
    out_tab = np.zeros(shape=(clus, len(out_names)))
    out_tab = pd.DataFrame(data=out_tab, columns=out_names)
            # for each cluster, we find the most common features
            # and the claims frequency per cluster
            
    for k in range(0, clus):
        idx = (X_clus == k)
        SA = Counter(y_test['satisfaction'][idx])[1]/y_test[idx].shape[0]
        G = Counter(X_test['Gender'][idx]).most_common(1)
        CT = Counter(X_test['Customer Type'][idx]).most_common(1)
        TT = Counter(X_test['Type of Travel'][idx]).most_common(1)
        C = Counter(X_test['Class'][idx]).most_common(1)
        IWS = Counter(X_test['Inflight wifi service'][idx]).most_common(1)
        DTC = Counter(X_test['Departure/Arrival time convenient'][idx]).most_common(1)
        EOB = Counter(X_test['Ease of Online booking'][idx]).most_common(1)
        GL = Counter(X_test['Gate location'][idx]).most_common(1)
        FD = Counter(X_test['Food and drink'][idx]).most_common(1)
        OB = Counter(X_test['Online boarding'][idx]).most_common(1)
        SC = Counter(X_test['Seat comfort'][idx]).most_common(1)
        IE = Counter(X_test['Inflight entertainment'][idx]).most_common(1)
        OS = Counter(X_test['On-board service'][idx]).most_common(1)
        LRS = Counter(X_test['Leg room service'][idx]).most_common(1)
        BG = Counter(X_test['Baggage handling'][idx]).most_common(1)
        CS = Counter(X_test['Checkin service'][idx]).most_common(1)
        IS = Counter(X_test['Inflight service'][idx]).most_common(1)
        CL = Counter(X_test['Cleanliness'][idx]).most_common(1)
        AG = Counter(X_test['AgeCat'][idx]).most_common(1)
        FDI = Counter(X_test['FlightDistanceCat'][idx]).most_common(1)
        DD = Counter(X_test['DepartDelayCat'][idx]).most_common(1)
        AD = Counter(X_test['ArrivalDelayCat'][idx]).most_common(1)
        out_tab.loc[k] = [G[0][0], CT[0][0], TT[0][0], C[0][0], IWS[0][0], DTC[0][0], EOB[0][0], GL[0][0],
                          FD[0][0], OB[0][0], SC[0][0], IE[0][0], OS[0][0], LRS[0][0], BG[0][0], CS[0][0], IS[0][0], CL[0][0],
                          AG[0][0], FDI[0][0], DD[0][0], AD[0][0], SA]

    ypred = out_tab['satisfaction'][X_clus]
    logloss = log_loss(y_test, ypred)
    return(logloss, out_tab)

#%% Plot crossentropy for clusters from 1 to 14

clus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] 
ce_kmeans = []
for k in clus : 
    res = burt(k)
    print(res)
    ce_kmeans.append(res)

#results obtained when using the vae for clustering form 1 to 14 clusters
ce_vae_kmeans = [0.68453617, 0.55606856, 0.55044388, 0.55590456, 0.44585187,0.42257368, 0.44059175, 0.43752365, 0.4272282 , 0.4081295 , 0.44631258, 0.4363611 , 0.42261589, 0.43909108]    
plt.title('Cross entropy : Burt vs VAE')
plt.plot(clus, ce_kmeans, '.-', label = "K-means Burt encoding")
plt.plot(clus, ce_vae_kmeans, '.-', label = "K-means compressed with VAE")
plt.legend()
plt.show()

#%% Results for 10 clusters

res = burt(10)
logloss = res[0]
res[1].mean(axis = 0)

