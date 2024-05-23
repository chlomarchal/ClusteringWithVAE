from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import statsmodels.api as sm

#############################################
# CHANGE PATH !!!!
#############################################

path = 'C:/Users/chloe/OneDrive/Documents/DATS2M_1/MEMOIRE/Code/code officiel/data'

#############################################
# Importing datasets and data preprocessing
#############################################

data = pd.read_csv(path + "/train.csv", sep=',')
data2 = pd.read_csv(path + "/test.csv", sep=',')
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


##############################################################################
# Fine-tuning VAE hyperparameters via custom grid search for kmeans and glm
##############################################################################

def vaefunction (intermediate_dim, lr, latent_dim, batchsize, clus, epoch = 100): 
    
    import random
    random.seed(2023)
    tf.random.set_seed(2023)
    np.random.seed(2023)

    scores1 = []
    scores2 = []
    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(X_one_hot_Complete)):
        train_fold = np.take(X_one_hot_Complete, train_index, axis = 0)
        test_fold = np.take(X_one_hot_Complete, test_index, axis = 0)
        y_train_fold = np.take(Y, train_index, axis = 0)
        y_test_fold = np.take(Y, test_index, axis = 0)
        
        original_dim = 113
                
        # encoder model
        inputs = keras.Input(shape=(original_dim,))
        d1_encoder = layers.Dense(intermediate_dim, activation="relu")(inputs)
        d2_encoder = layers.Dense(20, activation="relu")(d1_encoder)
        d3_encoder = layers.Dense(15, activation="relu")(d2_encoder)
        z_mean = layers.Dense(latent_dim)(d3_encoder)
        z_log_var = layers.Dense(latent_dim)(d3_encoder)
        encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")


        class Sampler(layers.Layer):
            def call(self, z_mean, z_log_var):
                batch_size = tf.shape(z_mean)[0]
                z_size = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch_size, z_size))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        #decoder model
        latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
        d1_decoder = layers.Dense(15, activation="relu")(latent_inputs)
        d2_decoder = layers.Dense(intermediate_dim, activation="relu")(d1_decoder)
        outputs = layers.Dense(original_dim, activation="sigmoid")(d2_decoder)
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
        vae.fit(train_fold, epochs=epoch, batch_size = batchsize, verbose = 0)
        
        
        encoded = vae.encoder.predict(test_fold)
        z = Sampler(name="z")(encoded[0], encoded[1])
        decoded = vae.decoder.predict(z)
        
        bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        loss = bce(test_fold, decoded)
        
        train_encoded = vae.encoder.predict(train_fold)
        z_train = Sampler(name="z")(train_encoded[0], train_encoded[1])
        train_compressed = z_train._numpy()
        
        test_encoded = vae.encoder.predict(test_fold)
        z_test = Sampler(name="z")(test_encoded[0], test_encoded[1])
        test_compressed = z_test._numpy()
        
        
        #Clustering
        n_clus = clus
        km = KMeans(init="random",
                    n_clusters=n_clus,
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
        out_tab = np.zeros(shape=(n_clus, len(out_names)))
        out_tab = pd.DataFrame(data=out_tab, columns=out_names)
        
        data_sub = np.take(data, test_index, axis = 0)
        X_sub = np.take(X, test_index, axis = 0)

        # for each cluster, we find the most common features
        for k in range(0, n_clus):
            idx = (X_clus == k)
            SA = Counter(data_sub['satisfaction'][idx])[1] / \
                data_sub['satisfaction'][idx].shape[0]
            G = Counter(X_sub['Gender'][idx]).most_common(1)
            CT = Counter(X_sub['Customer Type'][idx]).most_common(1)
            TT = Counter(X_sub['Type of Travel'][idx]).most_common(1)
            C = Counter(X_sub['Class'][idx]).most_common(1)
            IWS = Counter(X_sub['Inflight wifi service'][idx]).most_common(1)
            DTC = Counter(X_sub['Departure/Arrival time convenient']
                          [idx]).most_common(1)
            EOB = Counter(X_sub['Ease of Online booking'][idx]).most_common(1)
            FD = Counter(X_sub['Food and drink'][idx]).most_common(1)
            GL = Counter(X_sub['Gate location'][idx]).most_common(1)
            OB = Counter(X_sub['Online boarding'][idx]).most_common(1)
            SC = Counter(X_sub['Seat comfort'][idx]).most_common(1)
            IE = Counter(X_sub['Inflight entertainment'][idx]).most_common(1)
            OS = Counter(X_sub['On-board service'][idx]).most_common(1)
            LRS = Counter(X_sub['Leg room service'][idx]).most_common(1)
            BG = Counter(X_sub['Baggage handling'][idx]).most_common(1)
            CS = Counter(X_sub['Checkin service'][idx]).most_common(1)
            IS = Counter(X_sub['Inflight service'][idx]).most_common(1)
            CL = Counter(X_sub['Cleanliness'][idx]).most_common(1)
            AG = Counter(X_sub['AgeCat'][idx]).most_common(1)
            FDC = Counter(X_sub['FlightDistanceCat'][idx]).most_common(1)
            DD = Counter(X_sub['DepartDelayCat'][idx]).most_common(1)
            AD = Counter(X_sub['ArrivalDelayCat'][idx]).most_common(1)
            out_tab.loc[k] = [G[0][0], CT[0][0], TT[0][0], C[0][0], IWS[0][0], DTC[0][0], EOB[0][0], GL[0][0],
                              FD[0][0], OB[0][0], SC[0][0], IE[0][0], OS[0][0], LRS[0][0], BG[0][0], CS[0][0], IS[0][0], CL[0][0],
                              AG[0][0], FDC[0][0], DD[0][0], AD[0][0], SA]


        ypred = out_tab['satisfaction'][X_clus]
        logloss_k = log_loss(y_test_fold, ypred)
        scores1.append(logloss_k)
           
        
        #Regression
        glm_binom = sm.GLM(y_train_fold, train_compressed, family = sm.families.Binomial())
        res = glm_binom.fit()        
        ypred = res.predict(test_compressed)
        logloss_g = log_loss(y_test_fold, ypred)
        scores2.append(logloss_g)
       
    return (np.mean(scores1),np.mean(scores2), out_tab)

inte = [20, 30, 40] 
lr = [0.0001, 0.001]
lat = [5, 10, 15] #[5, 10, 15]
bat = [200, 500, 1000, 10000]
c = 10 #, 15]
epo = 100

#loops for grid searching
scores = []
for i in inte : 
    for l in lr : 
        for j in lat : 
            for k in bat : 
                    res = vaefunction(i, l, j, k, c, epo)
                    print("%f, %f with: %r, %r, %r, %r, %r, %r" % (res[1], res[0], i, l, j, k, c, epo))
                    scores.append([res[1], res[0],res[2], i, l, j, k, c, epo])
                    

######################################################################
# Plot cross-entropy of the GLM in function of the number of epochs
######################################################################

def vaefunctionglm (intermediate_dim, lr, latent_dim, batchsize, epoch): 
    import random
    random.seed(2023)
    tf.random.set_seed(2023)
    np.random.seed(2023)

    original_dim = 113
                    
    # encoder model
    inputs = keras.Input(shape=(original_dim,))
    d1_encoder = layers.Dense(intermediate_dim, activation="relu")(inputs)
    d2_encoder = layers.Dense(20, activation="relu")(d1_encoder)
    d3_encoder = layers.Dense(15, activation="relu")(d2_encoder)
    z_mean = layers.Dense(latent_dim)(d3_encoder)
    z_log_var = layers.Dense(latent_dim)(d3_encoder)
    encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")


    class Sampler(layers.Layer):
        def call(self, z_mean, z_log_var):
            batch_size = tf.shape(z_mean)[0]
            z_size = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch_size, z_size))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    #decoder model
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    d1_decoder = layers.Dense(15, activation="relu")(latent_inputs)
    d2_decoder = layers.Dense(intermediate_dim, activation="relu")(d1_decoder)
    outputs = layers.Dense(original_dim, activation="sigmoid")(d2_decoder)
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
    return(logloss)
   # print(res.summary())
   

epoch = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
ce_glm = []
for e in epoch : 
    res = vaefunctionglm(30, 0.001, 15, 1000, e)
    print(res)
    ce_glm.append(res)
    
#epoch = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
#test2 = test + test
#test3 = test2[0:15]
#test = test2[0:15]
plt.title('Cross entropy')
plt.plot(epoch, ce_glm, 'b.-')                    
                    

######################################################################
# Plot cross-entropy of the K-means in function of the number of epochs
######################################################################

def vaefunctionkmeans (intermediate_dim, lr, latent_dim, batchsize, clus, epoch = 100): 
    
    import random
    random.seed(2023)
    tf.random.set_seed(2023)
    np.random.seed(2023)
      
    original_dim = 113
    
    # encoder model
    inputs = keras.Input(shape=(original_dim,))
    d1_encoder = layers.Dense(intermediate_dim, activation="relu")(inputs)
    d2_encoder = layers.Dense(20, activation="relu")(d1_encoder)
    d3_encoder = layers.Dense(15, activation="relu")(d2_encoder)
    z_mean = layers.Dense(latent_dim)(d3_encoder)
    z_log_var = layers.Dense(latent_dim)(d3_encoder)
    encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")


    class Sampler(layers.Layer):
        def call(self, z_mean, z_log_var):
            batch_size = tf.shape(z_mean)[0]
            z_size = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch_size, z_size))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    #decoder model
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    d1_decoder = layers.Dense(15, activation="relu")(latent_inputs)
    d2_decoder = layers.Dense(intermediate_dim, activation="relu")(d1_decoder)
    outputs = layers.Dense(original_dim, activation="sigmoid")(d2_decoder)
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
    #opt = keras.optimizers.Adam(learning_rate=0.01)
    opt = keras.optimizers.RMSprop(learning_rate=lr)
    vae.compile(optimizer=opt, metrics=[keras.metrics.BinaryCrossentropy()])
    vae.fit(X_train, epochs=epoch, batch_size = batchsize, verbose = 0)
    
    encoded = vae.encoder.predict(X_test)
    z = Sampler(name="z")(encoded[0], encoded[1])
    decoded = vae.decoder.predict(z)
    
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    loss = bce(X_test, decoded)
    print("Binary Cross-Entropy Loss (TensorFlow):", loss.numpy())
    
    
    train_encoded = vae.encoder.predict(X_train)
    z_train = Sampler(name="z")(train_encoded[0], train_encoded[1])
    train_compressed = z_train._numpy()
    
    test_encoded = vae.encoder.predict(X_test)
    z_test = Sampler(name="z")(test_encoded[0], test_encoded[1])
    test_compressed = z_test._numpy()
       
    km = KMeans(init="random",
                        n_clusters=clus,
                        n_init=20,
                        max_iter=300,
                        random_state=42)  # seed of the rnd number generator
    km.fit(train_compressed)
            
    X_clus = km.predict(test_compressed)

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

    from sklearn.metrics import log_loss
    ypred = out_tab['satisfaction'][X_clus]
    yobs = y_test
    logloss = log_loss(yobs, ypred)
    return (logloss)


epoch = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
ce_kmeans = []
for e in epoch : 
    res = vaefunctionkmeans(40, 0.001, 5, 1000, 10, e)
    print(res)
    ce_kmeans.append(res)
    
plt.title('Cross entropy')
plt.plot(epoch, ce_kmeans, 'b.-')


######################################################################
# Plot cross-entropy of the K-means in function of the number of clusters
######################################################################

import random
random.seed(2023)
tf.random.set_seed(2023)
np.random.seed(2023)

lr = 0.001
intermediate_dim = 40
latent_dim = 5
batchsize = 1000 
epoch = 300 
clus = 10
original_dim = 113

# encoder model
inputs = keras.Input(shape=(original_dim,))
d1_encoder = layers.Dense(intermediate_dim, activation="relu")(inputs)
d2_encoder = layers.Dense(20, activation="relu")(d1_encoder)
d3_encoder = layers.Dense(15, activation="relu")(d2_encoder)
z_mean = layers.Dense(latent_dim)(d3_encoder)
z_log_var = layers.Dense(latent_dim)(d3_encoder)
encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")


class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#decoder model
latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
d1_decoder = layers.Dense(15, activation="relu")(latent_inputs)
d2_decoder = layers.Dense(intermediate_dim, activation="relu")(d1_decoder)
outputs = layers.Dense(original_dim, activation="sigmoid")(d2_decoder)
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

plt.title('Total loss')
plt.plot(history.history['total_loss'], color='blue', label='train')          
        
encoded = vae.encoder.predict(X_test)
z = Sampler(name="z")(encoded[0], encoded[1])
decoded = vae.decoder.predict(z)
        
bce = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
loss = bce(X_test, decoded)
        
train_encoded = vae.encoder.predict(X_train)
z_train = Sampler(name="z")(train_encoded[0], train_encoded[1])
train_compressed = z_train._numpy()

test_encoded = vae.encoder.predict(X_test)
z_test = Sampler(name="z")(test_encoded[0], test_encoded[1])
test_compressed = z_test._numpy()

ce2 = [] 
for clus in range (1, 15):    
    km = KMeans(init="random",
                    n_clusters=clus,
                    n_init=20,
                    max_iter=300,
                    random_state=42)  # seed of the rnd number generator
    km.fit(train_compressed)
        
    X_clus = km.predict(test_compressed)
            # goodness of fit
    print('K-means inertia {}'.format(km.inertia_))
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
    for k in range(0, clus):
        idx = (X_clus == k)
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


    ypred = out_tab['satisfaction'][X_clus]
    yobs = y_test
    logloss = log_loss(yobs, ypred)
    print(logloss)
    ce2.append([int(clus), logloss]) 

ce2= np.array(ce2).reshape(-1,2) 
plt.title('Cross entropy')
plt.plot(ce2[:, 0], ce2[:, 1], 'b.-')
