import numpy as np

class MyLogisticRegression:

    def __init__(self, learning_rate=0.001, C=1.0, n_iters=1000, batch_size=32):
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.batch_size = batch_size # rozmiar partii
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, x):
        # funkcja sigmoidalna przekształca wynik na prawdopodobieńswto, a wiec przedzial (0,1)
        return 1 / (1 + np.exp(-x))

    def _binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # obliczanie funkcji straty

    def _compute_loss(self, y_true, y_pred):
        # tutaj obliczana jest strata modelu z uwzględnieniem hiperparametru C (im większe C tym mniejsza kara za zbyt duze wagi)
        regularization_term = 1 / (2 * self.C) * np.sum(self.weights ** 2)
        return self._binary_cross_entropy(y_true, y_pred) + regularization_term

    def _gradient(self, X_batch, y_batch, class_label):
        # wyznaczanie gradientu funkcji straty względem wag modelu
        m = X_batch.shape[0]
        y_pred = self._sigmoid(np.dot(X_batch, self.weights[class_label]) + self.bias[class_label])
        dw = (1 / m) * np.dot(X_batch.T, (y_pred - y_batch))
        db = (1 / m) * np.sum(y_pred - y_batch)
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        for class_label in range(n_classes):
            binary_y = (y == class_label).astype(int)

            # przez określoną liczbę iteracji dobiera się wagi do cech
            for _ in range(self.n_iters):
                indices = np.random.choice(n_samples, self.batch_size, replace=True)
                X_batch, y_batch = X[indices], binary_y[indices] # losowe dane treningowe o wielkości batch_size

                dw, db = self._gradient(X_batch, y_batch, class_label)

                # wyznaczenie wartości dla liniowej kombinacji cech
                # z = w1*x1 + w2*x2 + ... + bias
                self.weights[class_label] -= self.lr * dw # w1, w2, ...
                self.bias[class_label] -= self.lr * db # bias

                y_pred = self._sigmoid(np.dot(X_batch, self.weights[class_label]) + self.bias[class_label])
                loss = self._compute_loss(y_batch, y_pred)
                self.losses.append(loss)


    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = self.weights.shape[0]

        y_pred = np.zeros((n_samples, n_classes))

        for class_label in range(n_classes):
            # przypisanie prawdopodobieństwa dla kazdej klasy
            y_pred[:, class_label] = self._sigmoid(np.dot(X, self.weights[class_label]) + self.bias[class_label])

        return np.argmax(y_pred, axis=1) # wybor klasy z najwiekszym prawdopodobienstwem

