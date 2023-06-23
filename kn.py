import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

def distancia_euclidiana(var1, var2):
    return np.sqrt(np.sum((var1 - var2) ** 2))

class KNN:
    #Inicializacion de clase, k representa al numero de vecinos cercanos
    def __init__(self, k=3):
        self.k = k

    #Datos de entrenamiento
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    #Prediccion de clase
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calcular las distancias entre el punto de prueba y todos los puntos de entrenamiento
        distancia = [distancia_euclidiana(x, x_train) for x_train in self.X_train]
        
        # Obtener las clases de los k vecinos más cercanos
        k_indices = np.argsort(distancia)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Realizar la predicción basada en la mayoría
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]    

# Datos de entrenamiento
X_train = np.array([[1.869423182, 2.791976048], [2.27318592, 2.477111136], [2.82734086, 1.893176098], [1.539694984, 2.50469424], [2.875420219, 2.221298127],
                    [2.092386085, 1.813369122], [3.693069637, 1.427792679], [1.576019048, 1.163440381], [2.009507082, 1.173333733], [0.8756594522, 0.6441210923],
                    [7.409944278, -0.3059050957], [8.435077921, -0.3707635751], [3.989400599, -0.949987061], [4.589165213, -0.03945974499], [4.766679469, -2.232500282], 
                    [5.953879833, -0.976158823], [5.426023982, -1.37631105], [1.984108012, -0.5616472507], [6.159202003, 1.423240065], [3.373001483, -0.9532170467]])
y_train = np.array(["A", "A","A","A","A","A","A","A","A","A","B","B","B","B","B","B","B","B","B","B"])

knn = KNN(k=3)
knn.fit(X_train, y_train)

X_test = np.array([[2.49856666, 2.295987279], [0.5060700405, 2.870605328], [2.821145579, 3.044439904], [1.103348767, 3.259222168], [2.178202675, 1.471900676],
                    [1.077220263, 2.603175495], [1.832200641, 2.745357961], [4.441079885, 2.683848879], [2.002112344, 1.410244628], [1.470189159, 2.31022513],
                    [2.430317441, -2.233903936], [4.173766546, 1.257892673], [7.21439806, -0.3865905836], [5.21280248, -3.246487248], [5.365220161, -3.660845623], 
                    [4.867157086, 0.6122608129], [5.186137477, -3.325490691], [4.002715986, -0.1184238242], [4.133560572, -2.589857889], [3.482855376, 0.5401176995]
])
predictions = knn.predict(X_test)

print("Datos de entrenamiento:\n", X_train, "\n", y_train, "\n")
print("Datos de predeicción:\n", X_test, "\n", predictions, "\n")

y_true = y_train
y_pred = predictions

matriz_confusion = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:\n",matriz_confusion)

vn, fp, fn, vp = matriz_confusion.ravel()

accuracy = (vp + vn) / (vp + fp + vn + fn)
print("Exactitud(Acuracy):",accuracy)

precision = vp/(vp + fp)
print("Precision:", precision)

sensitivity = vp / (vp + fn)
print("Sensibilidad:", sensitivity)

specificity = vn / (vn + fp)
print("Especificidad:", specificity)

f1 = (2 * precision * sensitivity) / (precision  + sensitivity)
print("Métrica F:", f1)
