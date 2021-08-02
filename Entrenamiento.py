from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
import os
import pickle


# cargar conjunto de datos RAVDESS
X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# imprimir algunos detalles
# número de muestras en los datos de entrenamiento
print("[+] Numero de muestras:", X_train.shape[0])
# número de muestras en los datos de prueba
print("[+] Numero de testeos:", X_test.shape[0])
# número de funciones utilizadas
# este es un vector de características extraídas
# usar el método utils.extract_features()
print("[+] Features:", X_train.shape[1])
# mejor modelo, determinado por una búsqueda de cuadrícula

model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 1000,
}

# inicializar multicapa Perceptron classifier
# con los mejores parámetros ( so far )
model = MLPClassifier(**model_params)

# Entrenar el modelo
print("[*] Entrenando...")
model.fit(X_train, y_train)

# predecir el 25% de los datos para medir qué tan buenos somos

y_pred = model.predict(X_test)
# calcular la precisión
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Precisión: {:.2f}%".format(accuracy*100))

# Ahora guardamos el modelo
# hacer directorio de resultados si aún no existe
if not os.path.isdir("result"):
    os.mkdir("result")
pickle.dump(model, open("result/mlp_classifier.model", "wb"))
