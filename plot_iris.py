from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

# Caricamento del dataset
iris = load_iris()
X = iris.data
y = iris.target

# Visualizzazione della distribuzione delle classi
plt.figure(figsize=(12, 4))

# Plot per i sepali
plt.subplot(2, 2, 1)
for i in range(3):
    mask = y == i
    plt.scatter(X[mask, 0], X[mask, 1], label=iris.target_names[i])
plt.xlabel('Lunghezza sepalo')
plt.ylabel('Larghezza sepalo')
plt.title('Distribuzione dei Sepali')
plt.legend()

# Plot per i petali
plt.subplot(2, 2, 2)
for i in range(3):
    mask = y == i
    plt.scatter(X[mask, 2], X[mask, 3], label=iris.target_names[i])
plt.xlabel('Lunghezza petali')
plt.ylabel('Larghezza petali')
plt.title('Distribuzione dei petali')
plt.legend()
# Ora tocca a te! Prova a fare il plot dei petali!
plt.subplot(2, 2, 3)
for i in range(3):
    mask = y == i
    plt.scatter(X[mask, 0], X[mask, 2], label=iris.target_names[i])
plt.xlabel('Lunghezza sepalo')
plt.ylabel('Lunghezza petali')
plt.title('Distribuzione dei petali')
plt.legend()

plt.subplot(2, 2, 4)
for i in range(3):
    mask = y == i
    plt.scatter(X[mask, 1], X[mask, 3], label=iris.target_names[i])
plt.xlabel('Larghezza petali')
plt.ylabel('Larghezza sepalo')
plt.title('Distribuzione dei petali')
plt.legend()
# Poi fai commit e push

# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=21)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled)
print(f"minimo: {min(X_train_scaled[:,0])}, massimo: {max(X_train_scaled[:,0])}, media: {np.mean(X_train_scaled[:,0])}")
X_test_scaled = scaler.transform(X_test)
mlp=MLPClassifier(
    hidden_layer_sizes = (4,2),
    activation="tanh",
    random_state=33,
    max_iter=10000
)
mlp.fit(X_train_scaled, y_train)
y_predict = mlp.predict(X_test_scaled)
accuracy = np.mean(y_predict == y_test)
print(f"Accuratezza: {accuracy:.2f}")
print(f"test loss: {mlp.loss_}")
print(f"numero iterazioni: {mlp.n_iter_}")

nuovo_iris = [[5.0, 3.5, 1.5, 0.2]]
nuovo_iris_scaled = scaler.transform(nuovo_iris)
previsione_iris = mlp.predict(nuovo_iris_scaled)
print("il mio nuovo fiore è.....")
print("...rullo di tamburi...")
print(iris.target_names[previsione_iris[0]])


