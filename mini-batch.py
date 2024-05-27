import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=100, batch_size=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.dw_history = []
        self.db_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                y_predicted = np.dot(X_batch, self.weights) + self.bias

                # Funkcja kosztu
                cost = np.mean((y_predicted - y_batch) ** 2)
                self.cost_history.append(cost)

                # Gradienty
                dw = (1 / self.batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / self.batch_size) * np.sum(y_predicted - y_batch)
                self.dw_history.append(dw)
                self.db_history.append(db)

                # Akt wag
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Przykładowe użycie
if __name__ == "__main__":
    # Dane test
    X_train = np.array([[1], [2], [3], [4], [5], [6]])
    y_train = np.array([2, 4, 6, 8, 10, 12])

    # Trening
    model = LinearRegression(learning_rate=0.01, n_iterations=100, batch_size=2)
    model.fit(X_train, y_train)

    # Testowanie
    X_test = np.array([[7], [8]])
    predictions = model.predict(X_test)
    print("Predictions:", predictions)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.plot(X_test, predictions, color='red', label='Predictions (model)')
    plt.scatter(X_test, predictions, color='green', label='Predictions (computed)')
    plt.xlabel('X')
    plt.ylabel('y = 2* X')
    plt.title('Linear Regression Prediction')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(len(model.cost_history[:500])), model.cost_history[:500], marker='o', markersize=5, color='green', label='Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Progress')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(len(model.dw_history)), model.dw_history, marker='o', markersize=5, color='blue', label='dw')
    plt.plot(range(len(model.db_history)), model.db_history, marker='o', markersize=5, color='orange', label='db')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Values')
    plt.title('Gradient Values History')
    plt.legend()
    
    plt.tight_layout()
    plt.show()