import numpy as np


class Perceptron(object):
    def __init__(
        self, learning_rate: float, permissible_error: float, iterations: int
    ) -> None:
        self.__learning_rate = learning_rate
        self.__permissible_error = permissible_error
        self.__iterations = iterations
        self.__W = None
        self.__weights_history = []
        self.__error_rule = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__W = np.random.uniform(-2, 2, (X.shape[1] + 1))

        # Se inserta el bias
        X = np.c_[np.ones(X.shape[0]), X]

        for _ in range(self.__iterations):
            u = np.dot(X, self.__W)
            y_cal = self.__step(u)
            error = y - y_cal
            delta_W = self.__learning_rate * np.dot(X.T, error)
            self.__W += delta_W

            error_norm = np.linalg.norm(error)
            self.__weights_history.append(self.__W.tolist())
            self.__error_rule.append(error_norm)

            if np.all(np.abs(error) <= self.__permissible_error):
                break

    def get_learning_rate(self) -> float:
        return self.__learning_rate

    def get_permissible_error(self) -> float:
        return self.__permissible_error

    def get_iterations(self) -> int:
        return self.__iterations

    def get_W(self) -> np.ndarray:
        return self.__W

    def get_weights_history(self) -> list:
        return self.__weights_history

    def get_errors_history(self) -> list:
        return self.__error_rule

    def __step(self, u: np.ndarray) -> np.ndarray:
        return np.where(u >= 0, 1, 0)
