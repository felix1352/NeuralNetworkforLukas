import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#Modell definieren

#Daten einlesen
l = 2
m = 1
g = 9.81
b = 0.7
T = 10  #simulation Time
dt = 0.1  # Schrittweite für numerisches Lösen
time = 10*T

def f(t, x, M):
    phi = x[0]
    y = x[1]
    dphidt = y
    dydt = 1/l*(-b*y-g*np.sin(phi)+M)
    return np.array([dphidt,dydt])

def berechne_winkel(phi0, M):
    t = 0
    phi = phi0[0]
    y = phi0[1]
    erg = np.array([])
    counter = 0
    while t < T:
        k1 = dt * f(t, [phi, y], M[counter])
        k2 = dt * f(t + dt / 2, [phi + k1[0] / 2, y + k1[1] / 2], M[counter])
        k3 = dt * f(t + dt / 2, [phi + k2[0] / 2, y + k2[1] / 2], M[counter])
        k4 = dt * f(t + dt, [phi + k3[0], y + k3[1]], M[counter])
        phi += (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y += (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        t += dt
        erg = np.append(erg, phi)
        counter += 1
    return np.reshape(erg, (erg.shape[0], 1))



#Trainingsdaten generieren

def generiere_trainingsdaten(startwinkel):
    trainX = []
    output_train = []
    tM = np.arange(0, T+dt, dt)
    for phi0_train in startwinkel:
        # Generiere Datenverlauf für den aktuellen Startwert
        in_amplitude = np.random.uniform(0, 2)
        in_frequency = np.random.uniform(0,2)
        M = in_amplitude*np.sin(in_frequency*tM)
        input_seq = M
        #input_seq = np.zeros([T*10, 1], dtype=int)
        input_seq = np.reshape(input_seq, (input_seq.shape[0], 1))
        tmp_train = np.concatenate((input_seq, np.zeros(shape=(input_seq.shape[0], 1))), axis=1)
        tmp_train = np.concatenate((phi0_train.T, tmp_train), axis=0)
        current_trainX = np.reshape(tmp_train[:-1], (1, tmp_train.shape[0]-1, tmp_train.shape[1]))
        current_output_train = berechne_winkel(phi0_train, M)
        current_output_train = np.reshape(current_output_train, (1, current_output_train.shape[0] , 1))
        current_output_train = np.expand_dims(current_output_train, axis=-1)
        # Füge die generierten Daten zum Gesamtdatensatz hinzu
        trainX.append(current_trainX)
        output_train.append(current_output_train)
    # Verkette die Daten entlang der 0-Achse, um eine einzige Numpy-Array-Darstellung zu erhalten
    trainX = np.concatenate(trainX, axis=0)
    output_train = np.concatenate(output_train, axis=0)
    return trainX, output_train

num_samples=1000
startwerte = np.random.rand(num_samples, 2, 1)
trainX, output_train = generiere_trainingsdaten(startwerte)

#Testdaten für eine Stichprobe zum plotten generieren
ttest = np.arange(0,T+dt, dt)
M = np.sin(ttest)
input_seq_test = M
#input_seq_test = np.zeros([T*10,1], dtype=int)
x0_test=np.random.rand(2,1)
input_seq_test=np.reshape(input_seq_test,(input_seq_test.shape[0],1))
tmp_test = np.concatenate((input_seq_test, np.zeros(shape=(input_seq_test.shape[0],1))), axis=1)
tmp_test = np.concatenate((x0_test.T, tmp_test), axis=0)
testX=np.reshape(tmp_test[:-1], (1,tmp_test.shape[0]-1,tmp_test.shape[1]))


output_test = berechne_winkel(x0_test, M)


#Hyperparamteroptimierung
import optuna

# Modell definieren
def create_model(trial):
    num_layers = trial.suggest_int('num_layers', 1, 3)  # Anzahl der Schichten
    num_neurons = trial.suggest_int('num_neurons', 32, 128)  # Anzahl der Neuronen pro Schicht

    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            # Erste Schicht mit Input-Shape
            model.add(SimpleRNN(num_neurons, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        elif i == num_layers - 1:
            # Letzte Schicht ohne return_sequences=True
            model.add(SimpleRNN(num_neurons, activation='relu'))
        else:
            # Zwischenschichten
            model.add(SimpleRNN(num_neurons, activation='relu', return_sequences=True))

    model.add(Dense(1))

    return model

# Training und Validierung mit den optimierten Hyperparametern
def train_and_evaluate_model(trial):
    model = create_model(trial)

    # Modell kompilieren
    model.compile(optimizer='adam', loss='mse')

    # Modell trainieren
    history = model.fit(trainX, output_train, epochs=10, batch_size=1, validation_split=0.3, verbose=0)

    # Rückgabe des Validierungsverlusts für die Hyperparameteroptimierung
    return history.history['val_loss'][-1]

# Optimierungsfunktion für Optuna
def objective(trial):
    val_loss = train_and_evaluate_model(trial)
    return val_loss

# Optuna-Studium erstellen und Hyperparameteroptimierung durchführen
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=100)

# Beste Hyperparameter und Validierungsverlust ausgeben
#print('Best Hyperparameters: ', study.best_params)
#print('Best Validation Loss: ', study.best_value)


#richtige Verlauf der Testdaten plotten
#time_plot=range(1,time+2)
#plt.subplot(1,2,1)
#plt.plot(time_plot,output_test,'r', label='Echter Verlauf')

#output_test = np.reshape(output_test,(1,output_test.shape[0],1))

#Model erstellen
#model = Sequential()
#model.add(SimpleRNN(79, activation='relu', input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
#model.add(Dense(1))

#Modell kompilieren
#model.compile(optimizer='adam', loss='mse')

#Modell trainieren  C:\Users\Felix Koch\Documents\BachelorArbeit\ModelsRnn
#filepath = "\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsRnn\\weights-{epoch:02d}-{val_loss:.6f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
#callbacks_list = [checkpoint]
#history = model.fit(trainX, output_train, epochs=10, batch_size=1, callbacks=callbacks_list, validation_split=0.3, verbose=2)

# load the model with the smallest validation loss
#model.load_weights("\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsRnn\\weights-05-0.000365.hdf5")


#Testen
#testPredict = model.predict(testX)

#Ergebnis der Prediction plotten
#plt.plot(time_plot,testPredict[0,:,0], label='Vorhergesagter Verlauf')

#plt.xlabel('Zeit in s')
#plt.ylabel('Auslenkung in rad')
#plt.legend()
#plt.subplot(1,2,2)
#plt.plot(time_plot, abs(testPredict[0,:,0]-output_test[0,:,0]))
#plt.xlabel('Zeit in s')
#plt.ylabel('Auslenkung in rad')
#plt.title('Absoluter Fehler')
#plt.show()

#model.save('RNN_Einfachpendel.h5')

#correlation, _ = pearsonr(testPredict[0,:,0], output_test[0,:,0])
#print('Korrelation: ',correlation)

#Plotten von trainingsloss und validationloss
#plt.figure()
#plt.plot(history.epoch,history.history['loss'], label='loss')
#plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
#plt.legend()
#plt.show()