# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:43:55 2024

@author: chloe
"""
from tensorflow import keras
from tensorflow.keras import layers
# Import numpy & matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import statsmodels.api as sm


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

# autoencoding, selection of inputs & outputs
Y = data[['satisfaction']]
X = data.drop(['Age', 'Flight Distance', 'Departure Delay in Minutes',
               'Arrival Delay in Minutes', 'satisfaction'], axis=1)

data_train, data_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2023, stratify=Y)

# Kereas works with array. Then we do "one-hot encoding"
X_one_hot_Complete = pd.get_dummies(X, drop_first=False)
X_train = pd.get_dummies(data_train, drop_first=False)
X_test = pd.get_dummies(data_test, drop_first=False)
#X_one_hot  = pd.get_dummies(X,drop_first=False)


# conversion as an array
X_train = np.asarray(X_train, dtype='float')
X_test = np.asarray(X_test, dtype='float')




#%% Final results of the GLM

import random
random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)


lr = 0.001
intermediate_dim = 30
latent_dim = 15
batchsize = 1000
epoch = 40
original_dim = 113
                
# encoder model
# encoder model
inputs = keras.Input(shape=(original_dim,))
m = layers.Dense(intermediate_dim, activation="relu")(inputs)
#m = layers.Dense(64, activation = "sigmoid")(m)
n = layers.Dense(20, activation="relu")(m)
h = layers.Dense(15, activation="relu")(n)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)
encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")


class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(15, activation="relu")(latent_inputs)
#x = layers.Dense(20, activation="relu")(x)
x = layers.Dense(intermediate_dim, activation="relu")(x)
#x = layers.Dense(64, activation = "sigmoid")(x)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, outputs, name="decoder")
        
        
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

        
        
vae = VAE(encoder, decoder)
opt = keras.optimizers.RMSprop(learning_rate=lr)
        #opt = "rmsprop"
vae.compile(optimizer=opt, metrics=[keras.metrics.BinaryCrossentropy()])
history = vae.fit(X_train, epochs=epoch, batch_size = batchsize, verbose = 1)

plt.title('Total loss')
plt.plot(history.history['total_loss'], color='blue', label='train')  
        
encoded = vae.encoder.predict(X_test)
z = Sampler(name="z")(encoded[0], encoded[1])
decoded = vae.decoder.predict(z)
        
bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
loss = bce(X_test, decoded)
#print("Binary Cross-Entropy Loss (TensorFlow):", loss.numpy())
        
encoder.summary()
decoder.summary()

train_encoded = vae.encoder.predict(X_train)
z_train = Sampler(name="z")(train_encoded[0], train_encoded[1])
train_compressed = z_train._numpy()

test_encoded = vae.encoder.predict(X_test)
z_test = Sampler(name="z")(test_encoded[0], test_encoded[1])
test_compressed = z_test._numpy()


glm_binom = sm.GLM(y_train, train_compressed, family = sm.families.Binomial())
res = glm_binom.fit()
ypred = res.predict(test_compressed)
logloss = log_loss(y_test, ypred)
print(logloss)
print(res.summary())

train = pd.DataFrame(train_compressed)

train_df_compressed = pd.DataFrame(train_compressed, columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15'])
plt.figure(figsize=(14,5))
sns.heatmap(train_df_compressed.corr(),annot=True,cmap='viridis',annot_kws={"size":12})


#%% Final results of the Kmeans

import random
random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)

lr = 0.001
intermediate_dim = 40
latent_dim = 5
batchsize = 1000 
epoch = 500 
clus = 10
original_dim = 113

# encoder model
inputs = keras.Input(shape=(original_dim,))
m = layers.Dense(intermediate_dim, activation="relu")(inputs)
#m = layers.Dense(64, activation = "sigmoid")(m)
n = layers.Dense(20, activation="relu")(m)
h = layers.Dense(15, activation="relu")(n)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)
encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")


class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(15, activation="relu")(latent_inputs)
#x = layers.Dense(20, activation="relu")(x)
x = layers.Dense(intermediate_dim, activation="relu")(x)
#x = layers.Dense(64, activation = "sigmoid")(x)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, outputs, name="decoder")
        
        
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
        
vae = VAE(encoder, decoder)
opt = keras.optimizers.RMSprop(learning_rate=lr)
        #opt = "rmsprop"
vae.compile(optimizer=opt, metrics=[keras.metrics.BinaryCrossentropy()])
history = vae.fit(X_train, epochs=epoch, batch_size = batchsize, verbose = 1)


plt.plot(history.history['total_loss'], color='blue', label='train')          
        
encoded = vae.encoder.predict(X_test)
z = Sampler(name="z")(encoded[0], encoded[1])
decoded = vae.decoder.predict(z)
        
bce = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
loss = bce(X_test, decoded)
#print("Binary Cross-Entropy Loss (TensorFlow):", loss.numpy())
        
train_encoded = vae.encoder.predict(X_train)
z_train = Sampler(name="z")(train_encoded[0], train_encoded[1])
train_compressed = z_train._numpy()

test_encoded = vae.encoder.predict(X_test)
z_test = Sampler(name="z")(test_encoded[0], test_encoded[1])
test_compressed = z_test._numpy()

train_df_compressed = pd.DataFrame(train_compressed, columns=['X1','X2','X3','X4','X5'])
plt.figure(figsize=(14,5))
sns.heatmap(train_df_compressed.corr(),annot=True,cmap='viridis',annot_kws={"size":15})

clus = 10

km = KMeans(init="random",
                    n_clusters=clus,
                    n_init=20,
                    max_iter=300,
                    random_state=42)  # seed of the rnd number generator
km.fit(train_compressed)
        
X_clus = km.predict(test_compressed)

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
    #freq = sum(data['NumberClaims'][idx])/sum(data['Duration'][idx])
    SA = Counter(y_test['satisfaction'][idx])[1] / \
        y_test['satisfaction'][idx].shape[0]
    G = Counter(data_test['Gender'][idx]).most_common(1)
    CT = Counter(data_test['Customer Type'][idx]).most_common(1)
    TT = Counter(data_test['Type of Travel'][idx]).most_common(1)
    C = Counter(data_test['Class'][idx]).most_common(1)
    IWS = Counter(data_test['Inflight wifi service'][idx]).most_common(1)
    DTC = Counter(data_test['Departure/Arrival time convenient'][idx]).most_common(1)
    EOB = Counter(data_test['Ease of Online booking'][idx]).most_common(1)
    GL = Counter(data_test['Gate location'][idx]).most_common(1)
    FD = Counter(data_test['Food and drink'][idx]).most_common(1)
    OB = Counter(data_test['Online boarding'][idx]).most_common(1)
    SC = Counter(data_test['Seat comfort'][idx]).most_common(1)
    IE = Counter(data_test['Inflight entertainment'][idx]).most_common(1)
    OS = Counter(data_test['On-board service'][idx]).most_common(1)
    LRS = Counter(data_test['Leg room service'][idx]).most_common(1)
    BG = Counter(data_test['Baggage handling'][idx]).most_common(1)
    CS = Counter(data_test['Checkin service'][idx]).most_common(1)
    IS = Counter(data_test['Inflight service'][idx]).most_common(1)
    CL = Counter(data_test['Cleanliness'][idx]).most_common(1)
    AG = Counter(data_test['AgeCat'][idx]).most_common(1)
    FDI = Counter(data_test['FlightDistanceCat'][idx]).most_common(1)
    DD = Counter(data_test['DepartDelayCat'][idx]).most_common(1)
    AD = Counter(data_test['ArrivalDelayCat'][idx]).most_common(1)
    out_tab.loc[k] = [G[0][0], CT[0][0], TT[0][0], C[0][0], IWS[0][0], DTC[0][0], EOB[0][0], GL[0][0],
                      FD[0][0], OB[0][0], SC[0][0], IE[0][0], OS[0][0], LRS[0][0], BG[0][0], CS[0][0], IS[0][0], CL[0][0],
                      AG[0][0], FDI[0][0], DD[0][0], AD[0][0], SA]


print(out_tab)

from sklearn.metrics import log_loss
ypred = out_tab['satisfaction'][X_clus]
yobs = y_test
logloss = log_loss(yobs, ypred)
print(logloss)

#mean values
out_tab.mean(axis = 0)

#%% TSNE plots 

X_one_hot_C = np.asarray(X_one_hot_Complete, dtype = "float")

X_encoded = vae.encoder.predict(X_one_hot_C)
z_X = Sampler(name="z")(X_encoded[0], X_encoded[1])
X_compressed = z_X._numpy()
Xclus = km.predict(X_compressed) 
Xclus = Xclus +1

tsne_full = TSNE(verbose = 1, random_state = 2023, perplexity=100, n_iter = 500).fit_transform(X_compressed)
df_tsne = pd.DataFrame(tsne_full, columns=['TSNE1', 'TSNE2'])
y = np.asarray(Y['satisfaction'])
df_tsne.insert(2,'satisfaction',y)
df_tsne.insert(3, 'cluster', Xclus)

df_tsne['satisfaction'] = df_tsne['satisfaction'].map(
    {0: 'neutral or dissatisfied', 1:'satisfied'})
fg = sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2',hue='satisfaction', palette='tab20',linewidth=0)
fg.legend(bbox_to_anchor= (1.2,-0.1))
plt.show()


plt.figure(figsize = (10, 7))
fg = sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2',hue='cluster', palette='tab20',linewidth=0)
                     #style = 'satisfaction', size = 200, markers = [".", "s"], edgecolor = "black")
fg.legend(bbox_to_anchor= (1.2,1))
plt.show()

#%%

#tsne_full3 = TSNE(n_components = 3, verbose = 1, random_state = 2023, perplexity=100, n_iter = 500).fit_transform(X_compressed)
#df_tsne3 = pd.DataFrame(tsne_full3, columns=['TSNE1', 'TSNE2', 'TSNE3'])
#df_tsne3.insert(3,'satisfaction',y)
#df_tsne3.insert(4, 'cluster', Xclus)

#%%

# Import libraries
#from mpl_toolkits import mplot3d
#import numpy as np
#import matplotlib.pyplot as plt


# Creating figure
#fig = plt.figure(figsize = (10, 7))
#ax = plt.axes(projection ="3d")

# Creating plot
#ax.scatter3D(df_tsne3.iloc[:,0], df_tsne3.iloc[:,1],df_tsne3.iloc[:,2], c = df_tsne3.iloc[:,4])

# show plot
#plt.show()


#%% GLM with dummyfied dataset

X_train_glm = pd.get_dummies(data_train, drop_first=True)
X_test_glm = pd.get_dummies(data_test, drop_first=True)
#X_one_hot  = pd.get_dummies(X,drop_first=False)

# Remove category xx_1 of variables where xx_0 is close to 0
X_dropped = X_train_glm.drop(['Inflight entertainment_1', 'On-board service_1', 
                              'Gate location_1', 'Inflight wifi service_1',
                                  'Checkin service_1', 'Inflight service_1', 
                                  'Seat comfort_1', 
                                  'Cleanliness_1'], axis = 1)
X_test_dropped = X_test_glm.drop(['Inflight entertainment_1', 'On-board service_1', 
                                  'Gate location_1', 'Inflight wifi service_1',
                                  'Checkin service_1', 'Inflight service_1', 
                                  'Seat comfort_1', 
                                  'Cleanliness_1'], axis = 1)

X_dropped = sm.add_constant(X_dropped)
glm_binom = sm.GLM(y_train, X_dropped, family = sm.families.Binomial(link=sm.genmod.families.links.Logit()))
res = glm_binom.fit()
print(res.summary())
X_test_dropped = sm.add_constant(X_test_dropped)
ypred = res.predict(X_test_dropped)
logloss = log_loss(y_test, ypred)
print(logloss)


