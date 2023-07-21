import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dynamics import get_dxdt
from scipy.integrate import solve_ivp
import optuna

dx_dt = get_dxdt()

#Modell definieren

#Daten einlesen

dt = 0.01
t_start = 0.0
t_end = int(10 + dt)

def F_func(t):
    return 0

default_parameter = {
        'g': 9.81,
        'M': 0.7,
        'm': 0.221,
        'l': 0.5,
        'b': 0.02,
        'c': 1
    }

t_eval = np.arange(t_start, t_end, dt) #Vektor der für gleichmäßige Zeitabstände beim Lösen sorgt, später wichtig wegen realer Abtastung

def berechne_verlauf(x0, F):
    # x0 muss nicht mehr modifiziert werden
    counter = 0
    x0 = [x0[0], 0, x0[1], 0]
    res = solve_ivp(lambda t, x: dx_dt(t, x, default_parameter, F=F_func(t)), [t_start, t_end], x0, t_eval=t_eval)
    erg = np.vstack([res.y[0], res.y[2]]).T
    return erg




#Trainingsdaten generieren

def generiere_trainingsdaten(startwerte):
    trainX = []
    output_train = []
    tF = np.arange(0, t_end + dt, dt)
    for x0 in startwerte:
        # Generiere Datenverlauf für den aktuellen Startwert
        F = 0
        input_seq = np.zeros([t_end * 100, 2], dtype=int)
        x0_2d = np.expand_dims(x0, axis=0)  # x0 in 2D-Array umwandeln
        tmp_train = input_seq
        tmp_train = np.concatenate((x0_2d, tmp_train), axis=0)  # x0 hat jetzt Shape (1, 2)
        current_trainX = np.reshape(tmp_train[:-1], (1, tmp_train.shape[0] - 1, tmp_train.shape[1]))
        current_output_train = berechne_verlauf(x0, F)  # berechne_verlauf benötigt 2D-Array
        current_output_train = np.reshape(current_output_train, (1, current_output_train.shape[0], 2, 1))
        # Füge die generierten Daten zum Gesamtdatensatz hinzu
        trainX.append(current_trainX)
        output_train.append(current_output_train)
    # Verkette die Daten entlang der 0-Achse, um eine einzige Numpy-Array-Darstellung zu erhalten
    trainX = np.concatenate(trainX, axis=0)
    output_train = np.concatenate(output_train, axis=0)
    return trainX, output_train





num_samples = 10
startwerte = np.ones((num_samples, 2), dtype=int)  # Shape als (num_samples, 2)

trainX, output_train = generiere_trainingsdaten(startwerte)

#Testdaten für eine Stichprobe zum plotten generieren
ttest = np.arange(0,t_end+dt, dt)

input_seq_test = np.zeros([t_end*100, 2], dtype=int)
#x0_test=np.random.rand(2,1)
x0_test = np.ones((2), dtype=int)
F = 0
x0_2d = np.expand_dims(x0_test, axis=0)  # x0 in 2D-Array umwandeln
tmp_train = input_seq_test
tmp_train = np.concatenate((x0_2d, tmp_train), axis=0)  # x0 hat jetzt Shape (1, 2)
testX = np.reshape(tmp_train[:-1], (1, tmp_train.shape[0] - 1, tmp_train.shape[1]))
output_test = berechne_verlauf(x0_test, F)


#Hyperparamteroptimierung


# Modell definieren

def create_model(trial):
    num_neurons = trial.suggest_int('num_neurons', 32, 128)  # Anzahl der Neuronen pro Schicht
    num_layers = trial.suggest_int('num_layers', 1, 6)  # Anzahl der Schichten

    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            # Erste Schicht mit Input-Shape
            model.add(SimpleRNN(num_neurons, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
                                return_sequences=True))
        elif i == num_layers - 1:
            # Letzte Schicht mit return_sequences=True, um eine Sequenz zurückzugeben
            model.add(SimpleRNN(num_neurons, activation='relu', return_sequences=True))
        else:
            # Zwischenschichten
            model.add(SimpleRNN(num_neurons, activation='relu', return_sequences=True))

    # Ausgabeschicht mit den richtigen Dimensionen (1000, 2, 1)
    model.add(Dense(1))  # 1 Ausgabeeinheit für die zusätzliche Dimension
    model.add(Dense(2))  # 2 Ausgabeeinheiten für die beiden Merkmale (Winkel und Position)

    return model


# Training und Validierung mit den optimierten Hyperparametern
def train_and_evaluate_model(trial):
    model = create_model(trial)

    # Modell kompilieren
    model.compile(optimizer='adam', loss='mse')

    # Modell trainieren
    history = model.fit(trainX, output_train, epochs=10, batch_size=100, validation_split=0.3, verbose=0)

    # Rückgabe des Validierungsverlusts für die Hyperparameteroptimierung
    return history.history['val_loss'][-1]

# Optimierungsfunktion für Optuna
def objective(trial):
    val_loss = train_and_evaluate_model(trial)
    return val_loss

# Optuna-Studium erstellen und Hyperparameteroptimierung durchführen
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=20)

# Beste Hyperparameter und Validierungsverlust ausgeben
#print('Best Hyperparameters: ', study.best_params)
#print('Best Validation Loss: ', study.best_value)

def convertToMatrix(data, step):
    X, Y=[],[]
    for i in range(len(data)-step):
        d=i+step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)


output_train = output_train.squeeze()
X = np.zeros((0,5,2))
Y = np.zeros((0,2))
for sample in output_train:
    X_sample, Y_sample = convertToMatrix(sample, 5)
    X = np.concatenate((X, X_sample), axis=0)
    Y = np.concatenate((Y, Y_sample), axis=0)


#Model erstellen
model = Sequential()
model.add(SimpleRNN(120, activation='relu', input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dense(10))
model.add(Dense(2))

#Modell kompilieren
model.compile(optimizer='adam', loss='mse')

#Modell trainieren  C:\Users\Felix Koch\Documents\BachelorArbeit\ModelsRnn
filepath = "\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsRnn\\weights-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
callbacks_list = [checkpoint]
history = model.fit(X, Y, epochs=10, batch_size=1, callbacks=callbacks_list, validation_split=0.3, verbose=2)
#model.fit(trainX, output_train, epochs=10, batch_size=10, validation_split=0.3)
# load the model with the smallest validation loss
#model.load_weights("\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsRnn\\weights-05-0.000365.hdf5")


#Testen
testPredict = model.predict(testX)

#Ergebnis der Prediction im Vergleich zum Echten Verlauf plotten
time_plot = np.arange(0, t_end, dt)
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(time_plot, output_test[:, 0], 'r', label='Echter Verlauf des Winkels')
plt.plot(time_plot, testPredict[0, :, 0], 'g', label='Vorhergesagter Verlauf des Winkels')
plt.xlabel('Zeit')
plt.ylabel('Winkel')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_plot, output_test[:, 1], 'b', label='Echter Verlauf der Position')
plt.plot(time_plot, testPredict[0, :, 1], 'm', label='Vorhergesagter Verlauf der Position')
plt.xlabel('Zeit')
plt.ylabel('Position')
plt.legend()

plt.show()



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
plt.figure()
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()