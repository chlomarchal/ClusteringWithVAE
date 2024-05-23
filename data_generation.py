# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:43:55 2024

@author: chloe
"""
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import statsmodels.api as sm
import seaborn as sns

#%% Preparation of the data

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

#Then we do "one-hot encoding"
X_one_hot_Complete = pd.get_dummies(X, drop_first=False)
X_train = pd.get_dummies(data_train, drop_first=False)
X_test = pd.get_dummies(data_test, drop_first=False)

# conversion as an array as Keras works with arrays
X_train = np.asarray(X_train, dtype='float')
X_test = np.asarray(X_test, dtype='float')

#%% VAE for data generation

import random
random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)

lr = 0.001
intermediate_dim = 30
latent_dim = 15
batchsize = 30
epoch = 40
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
vae.compile(optimizer=opt, metrics=[keras.metrics.BinaryCrossentropy()])
history = vae.fit(X_train, epochs=epoch, batch_size = batchsize, verbose = 1)
        
#%% Construct synthetic dataset

import random
random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)

generator_model = decoder

# sample latent_dim from independent N(0,1)
passengers = pd.DataFrame(tf.random.normal(shape=[X.shape[0], latent_dim],
                                          mean=np.zeros(latent_dim),
                                          stddev=np.full(latent_dim, 1), seed = 2023))

# synthetic tabular data
gen = generator_model.predict(passengers)
generated_passengers = pd.DataFrame(gen,columns=X_one_hot_Complete.columns)
generated_passengers.head()

def undummify(df, prefix_sep="_"):
    cols2collapse = {
       item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    # new dataframe
    series_list   = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            # we select the modality with the maximum probability
            undummified = (df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col))
            # we add the new undummified column
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

X_generated = undummify(generated_passengers, prefix_sep="_")

#%% one hot encoding of generated set for prediction of satisfaction using GLM
X_reconst_one_hot = pd.get_dummies(X_generated, drop_first=False)
X_reconst_one_hot.insert(27, 'Gate location_0', 0)
X_reconst_one_hot.insert(45, 'Seat comfort_0', 0)
X_reconst_one_hot.insert(51, 'Inflight entertainment_0', 0)
X_reconst_one_hot.insert(57, 'On-board service_0', 0)
X_reconst_one_hot.insert(74, 'Checkin service_0', 0)
X_reconst_one_hot.insert(80, 'Inflight service_0', 0)
X_reconst_one_hot.insert(86, 'Cleanliness_0', 0)
X_reconst_one_hot.insert(102, 'FlightDistanceCat_4001 - 5000', 0)
X_reconst_one_hot.insert(112, 'ArrivalDelayCat_241+', 0)
X_reconst_one_hot = X_reconst_one_hot[['Gender_Female', 'Gender_Male',
                                       'Customer Type_Loyal Customer', 'Customer Type_disloyal Customer',
                                       'Type of Travel_Business travel', 'Type of Travel_Personal Travel',
                                       'Class_Business', 'Class_Eco', 'Class_Eco Plus',
                                       'Inflight wifi service_0', 'Inflight wifi service_1', 'Inflight wifi service_2', 'Inflight wifi service_3', 'Inflight wifi service_4','Inflight wifi service_5',
                                       'Departure/Arrival time convenient_0', 'Departure/Arrival time convenient_1', 'Departure/Arrival time convenient_2', 'Departure/Arrival time convenient_3', 'Departure/Arrival time convenient_4', 'Departure/Arrival time convenient_5',
                                       'Ease of Online booking_0', 'Ease of Online booking_1', 'Ease of Online booking_2', 'Ease of Online booking_3', 'Ease of Online booking_4', 'Ease of Online booking_5',
                                       'Gate location_0', 'Gate location_1', 'Gate location_2', 'Gate location_3', 'Gate location_4', 'Gate location_5',
                                       'Food and drink_0', 'Food and drink_1', 'Food and drink_2', 'Food and drink_3', 'Food and drink_4', 'Food and drink_5',
                                       'Online boarding_0', 'Online boarding_1', 'Online boarding_2', 'Online boarding_3', 'Online boarding_4', 'Online boarding_5',
                                       'Seat comfort_0', 'Seat comfort_1', 'Seat comfort_2', 'Seat comfort_3', 'Seat comfort_4', 'Seat comfort_5',
                                       'Inflight entertainment_0', 'Inflight entertainment_1', 'Inflight entertainment_2', 'Inflight entertainment_3', 'Inflight entertainment_4', 'Inflight entertainment_5',
                                       'On-board service_0', 'On-board service_1', 'On-board service_2', 'On-board service_3', 'On-board service_4', 'On-board service_5',
                                       'Leg room service_0', 'Leg room service_1', 'Leg room service_2', 'Leg room service_3', 'Leg room service_4', 'Leg room service_5',
                                       'Baggage handling_1', 'Baggage handling_2', 'Baggage handling_3', 'Baggage handling_4', 'Baggage handling_5',
                                       'Checkin service_0', 'Checkin service_1', 'Checkin service_2', 'Checkin service_3', 'Checkin service_4', 'Checkin service_5',
                                       'Inflight service_0', 'Inflight service_1', 'Inflight service_2', 'Inflight service_3', 'Inflight service_4', 'Inflight service_5',
                                       'Cleanliness_0', 'Cleanliness_1', 'Cleanliness_2', 'Cleanliness_3', 'Cleanliness_4', 'Cleanliness_5',
                                       'AgeCat_7 - 18', 'AgeCat_19 - 30', 'AgeCat_31 - 40', 'AgeCat_41 - 50', 'AgeCat_51 - 64', 'AgeCat_65-85',
                                       'FlightDistanceCat_0 - 1000', 'FlightDistanceCat_1001 - 2000', 'FlightDistanceCat_2001 - 3000', 'FlightDistanceCat_3001 - 4000', 'FlightDistanceCat_4001 - 5000',
                                       'DepartDelayCat_0 - 5', 'DepartDelayCat_6 - 60', 'DepartDelayCat_61 - 120', 'DepartDelayCat_121 - 240', 'DepartDelayCat_240+',
                                       'ArrivalDelayCat_0 - 5', 'ArrivalDelayCat_6 - 60', 'ArrivalDelayCat_61 - 120', 'ArrivalDelayCat_121 - 240', 'ArrivalDelayCat_241+']]

#%% prediction of satisfaction using GLM

train_encoded = vae.encoder.predict(X_train)
z_train = Sampler(name="z")(train_encoded[0], train_encoded[1])
train_compressed = z_train._numpy()

gen_encoded = vae.encoder.predict(X_reconst_one_hot)
z_gen = Sampler(name="z")(gen_encoded[0], gen_encoded[1])
gen_compressed = z_gen._numpy()

#predict satisfaction using the glm
glm_binom = sm.GLM(y_train, train_compressed, family = sm.families.Binomial())
res = glm_binom.fit()
y_gen = res.predict(gen_compressed)

#distribution of the predicted satisfaction for the generated set
plt.hist(y_gen)

#0 or 1 for the satisfaction
X_reconstructed = pd.concat([X_generated, pd.DataFrame(y_gen, columns=['satisfaction'])], axis = 1)
X_reconstructed.loc[X_reconstructed['satisfaction'] >= 0.5, 'satisfaction'] = 1
X_reconstructed.loc[X_reconstructed['satisfaction'] < 0.5, 'satisfaction'] = 0


#%% one-hot encoding of the generated set with satisfaction for applying Kmeans

X_reconstructed_one_hot = pd.get_dummies(X_reconstructed, drop_first=False)
X_reconstructed_one_hot.insert(27, 'Gate location_0', 0)
X_reconstructed_one_hot.insert(45, 'Seat comfort_0', 0)
X_reconstructed_one_hot.insert(51, 'Inflight entertainment_0', 0)
X_reconstructed_one_hot.insert(57, 'On-board service_0', 0)
X_reconstructed_one_hot.insert(74, 'Checkin service_0', 0)
X_reconstructed_one_hot.insert(80, 'Inflight service_0', 0)
X_reconstructed_one_hot.insert(86, 'Cleanliness_0', 0)
X_reconstructed_one_hot.insert(102, 'FlightDistanceCat_4001 - 5000', 0)
X_reconstructed_one_hot.insert(112, 'ArrivalDelayCat_241+', 0)
X_reconstructed_one_hot = X_reconstructed_one_hot[['Gender_Female', 'Gender_Male',
                                       'Customer Type_Loyal Customer', 'Customer Type_disloyal Customer',
                                       'Type of Travel_Business travel', 'Type of Travel_Personal Travel',
                                       'Class_Business', 'Class_Eco', 'Class_Eco Plus',
                                       'Inflight wifi service_0', 'Inflight wifi service_1', 'Inflight wifi service_2', 'Inflight wifi service_3', 'Inflight wifi service_4','Inflight wifi service_5',
                                       'Departure/Arrival time convenient_0', 'Departure/Arrival time convenient_1', 'Departure/Arrival time convenient_2', 'Departure/Arrival time convenient_3', 'Departure/Arrival time convenient_4', 'Departure/Arrival time convenient_5',
                                       'Ease of Online booking_0', 'Ease of Online booking_1', 'Ease of Online booking_2', 'Ease of Online booking_3', 'Ease of Online booking_4', 'Ease of Online booking_5',
                                       'Gate location_0', 'Gate location_1', 'Gate location_2', 'Gate location_3', 'Gate location_4', 'Gate location_5',
                                       'Food and drink_0', 'Food and drink_1', 'Food and drink_2', 'Food and drink_3', 'Food and drink_4', 'Food and drink_5',
                                       'Online boarding_0', 'Online boarding_1', 'Online boarding_2', 'Online boarding_3', 'Online boarding_4', 'Online boarding_5',
                                       'Seat comfort_0', 'Seat comfort_1', 'Seat comfort_2', 'Seat comfort_3', 'Seat comfort_4', 'Seat comfort_5',
                                       'Inflight entertainment_0', 'Inflight entertainment_1', 'Inflight entertainment_2', 'Inflight entertainment_3', 'Inflight entertainment_4', 'Inflight entertainment_5',
                                       'On-board service_0', 'On-board service_1', 'On-board service_2', 'On-board service_3', 'On-board service_4', 'On-board service_5',
                                       'Leg room service_0', 'Leg room service_1', 'Leg room service_2', 'Leg room service_3', 'Leg room service_4', 'Leg room service_5',
                                       'Baggage handling_1', 'Baggage handling_2', 'Baggage handling_3', 'Baggage handling_4', 'Baggage handling_5',
                                       'Checkin service_0', 'Checkin service_1', 'Checkin service_2', 'Checkin service_3', 'Checkin service_4', 'Checkin service_5',
                                       'Inflight service_0', 'Inflight service_1', 'Inflight service_2', 'Inflight service_3', 'Inflight service_4', 'Inflight service_5',
                                       'Cleanliness_0', 'Cleanliness_1', 'Cleanliness_2', 'Cleanliness_3', 'Cleanliness_4', 'Cleanliness_5',
                                       'AgeCat_7 - 18', 'AgeCat_19 - 30', 'AgeCat_31 - 40', 'AgeCat_41 - 50', 'AgeCat_51 - 64', 'AgeCat_65-85',
                                       'FlightDistanceCat_0 - 1000', 'FlightDistanceCat_1001 - 2000', 'FlightDistanceCat_2001 - 3000', 'FlightDistanceCat_3001 - 4000', 'FlightDistanceCat_4001 - 5000',
                                       'DepartDelayCat_0 - 5', 'DepartDelayCat_6 - 60', 'DepartDelayCat_61 - 120', 'DepartDelayCat_121 - 240', 'DepartDelayCat_240+',
                                       'ArrivalDelayCat_0 - 5', 'ArrivalDelayCat_6 - 60', 'ArrivalDelayCat_61 - 120', 'ArrivalDelayCat_121 - 240', 'ArrivalDelayCat_241+', 'satisfaction']]

Yr = X_reconstructed_one_hot[['satisfaction']]
Xr = X_reconstructed_one_hot.drop(['satisfaction'], axis=1)

#%% VAE for Kmeans

import random
random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)

lr = 0.001
intermediate_dim = 40
latent_dim = 5
batchsize = 5000
epoch = 50
clus = 10
original_dim = 113

# encoder model
inputs = keras.Input(shape=(original_dim,))
d1_encoder = layers.Dense(intermediate_dim, activation="relu")(inputs)
#m = layers.Dense(64, activation = "sigmoid")(m)
d2_encoder = layers.Dense(20, activation="relu")(d1_encoder)
d3_encoder = layers.Dense(15, activation="relu")(d2_encoder)
z_mean = layers.Dense(latent_dim)(d3_encoder)
z_log_var = layers.Dense(latent_dim)(d3_encoder)
encoder_kmeans = keras.Model(inputs, [z_mean, z_log_var], name="encoder")

class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
d1_decoder = layers.Dense(15, activation="relu")(latent_inputs)
d2_decoder = layers.Dense(intermediate_dim, activation="relu")(d1_decoder)
outputs = layers.Dense(original_dim, activation="sigmoid")(d2_decoder)
decoder_kmeans = keras.Model(latent_inputs, outputs, name="decoder")
        
        
class VAE(keras.Model):
    def __init__(self, encoder_kmeans, decoder_kmeans, **kwargs):
        super().__init__(**kwargs)
        self.encoder_kmeans = encoder_kmeans
        self.decoder_kmeans = decoder_kmeans
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
            z_mean, z_log_var = self.encoder_kmeans(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder_kmeans(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction)))
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
        
vae_kmeans = VAE(encoder_kmeans, decoder_kmeans)
opt = keras.optimizers.RMSprop(learning_rate=lr)
vae_kmeans.compile(optimizer=opt, metrics=[keras.metrics.BinaryCrossentropy()])
history = vae_kmeans.fit(X_train, epochs=epoch, batch_size = batchsize, verbose = 1)


#%% Compression of the datasets
train_encoded = vae_kmeans.encoder_kmeans.predict(X_train)
z_train = Sampler(name="z")(train_encoded[0], train_encoded[1])
train_compressed = z_train._numpy()

X_gen_encoded = vae_kmeans.encoder_kmeans.predict(Xr)
z_generated = Sampler(name="z")(X_gen_encoded[0], X_gen_encoded[1])
X_gen_compressed = z_generated._numpy()

#%% Clustering with Kmeans

clus = 10

km = KMeans(init="random",
                    n_clusters=clus,
                    n_init=20,
                    max_iter=1000,
                    random_state=42)  # seed of the rnd number generator
km.fit(train_compressed)

#vector of clusters associated to each individual        
X_clus = km.predict(X_gen_compressed)

#out_tab : tables of dominant profiles in each cluster
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
# and the satisfaction probability for each cluster   
for k in range(0, clus):
    idx = (X_clus == k)
    SA = Counter(Yr['satisfaction'][idx])[1] / Yr['satisfaction'][idx].shape[0]
    G = Counter(X_reconstructed['Gender'][idx]).most_common(1)
    CT = Counter(X_reconstructed['Customer Type'][idx]).most_common(1)
    TT = Counter(X_reconstructed['Type of Travel'][idx]).most_common(1)
    C = Counter(X_reconstructed['Class'][idx]).most_common(1)
    IWS = Counter(X_reconstructed['Inflight wifi service'][idx]).most_common(1)
    DTC = Counter(X_reconstructed['Departure/Arrival time convenient'][idx]).most_common(1)
    EOB = Counter(X_reconstructed['Ease of Online booking'][idx]).most_common(1)
    GL = Counter(X_reconstructed['Gate location'][idx]).most_common(1)
    FD = Counter(X_reconstructed['Food and drink'][idx]).most_common(1)
    OB = Counter(X_reconstructed['Online boarding'][idx]).most_common(1)
    SC = Counter(X_reconstructed['Seat comfort'][idx]).most_common(1)
    IE = Counter(X_reconstructed['Inflight entertainment'][idx]).most_common(1)
    OS = Counter(X_reconstructed['On-board service'][idx]).most_common(1)
    LRS = Counter(X_reconstructed['Leg room service'][idx]).most_common(1)
    BG = Counter(X_reconstructed['Baggage handling'][idx]).most_common(1)
    CS = Counter(X_reconstructed['Checkin service'][idx]).most_common(1)
    IS = Counter(X_reconstructed['Inflight service'][idx]).most_common(1)
    CL = Counter(X_reconstructed['Cleanliness'][idx]).most_common(1)
    AG = Counter(X_reconstructed['AgeCat'][idx]).most_common(1)
    FDI = Counter(X_reconstructed['FlightDistanceCat'][idx]).most_common(1)
    DD = Counter(X_reconstructed['DepartDelayCat'][idx]).most_common(1)
    AD = Counter(X_reconstructed['ArrivalDelayCat'][idx]).most_common(1)
    out_tab.loc[k] = [G[0][0], CT[0][0], TT[0][0], C[0][0], IWS[0][0], DTC[0][0], EOB[0][0], GL[0][0],
                      FD[0][0], OB[0][0], SC[0][0], IE[0][0], OS[0][0], LRS[0][0], BG[0][0], CS[0][0], IS[0][0], CL[0][0],
                      AG[0][0], FDI[0][0], DD[0][0], AD[0][0], SA]


ypred = out_tab['satisfaction'][X_clus]
yobs = Yr
logloss = log_loss(yobs, ypred)
logloss
out_tab

#%% Barplots of generated dataset part 1

num_sub_plot=len(X_reconstructed.columns)
fig,ax=plt.subplots(4,3,figsize=(18,24))
col = X_reconstructed.columns
sns.countplot(data=X_reconstructed,x=col[0],ax=ax[0,0], color= '#2DA3CD')
sns.countplot(data=X_reconstructed,x=col[1],ax=ax[0,1], color= '#2DA3CD')
sns.countplot(data=X_reconstructed, x=col[2], color = '#2DA3CD', ax=ax[0,2], order = ['Business travel', 'Personal Travel'])
sns.countplot(data=X_reconstructed,x=col[3],ax=ax[1,0], color= '#2DA3CD')
sns.countplot(data=X_reconstructed,x=col[4],ax=ax[1,1], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed, x=col[5], color = '#2DA3CD', ax=ax[1,2], order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[6],ax=ax[2,0], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[7],ax=ax[2,1], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[8],ax=ax[2,2], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[9],ax=ax[3,0], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[10],ax=ax[3,1], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[11],ax=ax[3,2], color= '#2DA3CD', order = ['0','1','2','3','4','5'])


#%% Barplots of generated dataset part 2


num_sub_plot=len(data.columns)
fig,ax=plt.subplots(4,3,figsize=(18,24))
sns.countplot(data=X_reconstructed,x=col[12],ax=ax[0,0], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[13],ax=ax[0,1], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[14],ax=ax[0,2], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[15],ax=ax[1,0], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[16],ax=ax[1,1], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[17],ax=ax[1,2], color= '#2DA3CD', order = ['0','1','2','3','4','5'])
sns.countplot(data=X_reconstructed,x=col[18],ax=ax[2,0], color= '#2DA3CD', order = ["7 - 18", "19 - 30", "31 - 40", "41 - 50", "51 - 64", "65-85"])
sns.countplot(data=X_reconstructed,x=col[19],ax=ax[2,1], color= '#2DA3CD')
sns.countplot(data=X_reconstructed,x=col[20],ax=ax[2,2], color= '#2DA3CD')
sns.countplot(data=X_reconstructed,x=col[21],ax=ax[3,0], color= '#2DA3CD')
sns.countplot(data=X_reconstructed,x=col[22],ax=ax[3,1], color= '#2DA3CD')
ax[3,2].set_axis_off()
