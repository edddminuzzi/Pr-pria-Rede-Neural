"""Classificador feedforward do Iris implementado com NumPy e Pandas."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
"""Usando o sklearn para importar o DataSet Iris"""

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "iris.data"
RESULTS = ROOT / "resultado"
SEED = 42
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    df["target_id"] = iris.target

    nomes = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica",
    }

    df["target"] = df["target_id"].map(nomes)

    return df

def stratified_split(df, test_ratio=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(df["target_id"].to_numpy() == label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def one_hot(y, classes=3):
    out = np.zeros((len(y), classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


class DenseNetwork:
    """Rede 4 -> 8 -> 8 -> 3 com ReLU e softmax, treinada com Adam."""

    def __init__(self, input_dim=4, hidden_dims=(8, 8), output_dim=3, seed=SEED):
        rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_dims, output_dim]
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            # He para camadas ReLU; a escala também é adequada para a saída pequena.
            self.weights.append(rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(np.float64)

    @staticmethod
    def softmax(z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x):
        activations = [x]
        pre_acts = []
        current = x
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            current = self.relu(z)
            activations.append(current)
        z = current @ self.weights[-1] + self.biases[-1]
        pre_acts.append(z)
        probs = self.softmax(z)
        activations.append(probs)
        return probs, activations, pre_acts

    def predict(self, x):
        return np.argmax(self.forward(x)[0], axis=1)

    def loss_and_gradients(self, x, y_one_hot):
        probs, activations, pre_acts = self.forward(x)
        n = x.shape[0]
        loss = -np.sum(y_one_hot * np.log(probs + 1e-12)) / n
        grad = (probs - y_one_hot) / n
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            grad_w[layer] = activations[layer].T @ grad
            grad_b[layer] = np.sum(grad, axis=0, keepdims=True)
            if layer > 0:
                grad = (grad @ self.weights[layer].T) * self.relu_grad(pre_acts[layer - 1])
        return loss, grad_w, grad_b

    def fit(self, x, y, epochs=500, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        y_oh = one_hot(y)
        history = []
        for step in range(1, epochs + 1):
            loss, grad_w, grad_b = self.loss_and_gradients(x, y_oh)
            for i in range(len(self.weights)):
                self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grad_w[i]
                self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * grad_w[i] ** 2
                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grad_b[i]
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * grad_b[i] ** 2
                m_w_hat = self.m_w[i] / (1 - beta1 ** step)
                v_w_hat = self.v_w[i] / (1 - beta2 ** step)
                m_b_hat = self.m_b[i] / (1 - beta1 ** step)
                v_b_hat = self.v_b[i] / (1 - beta2 ** step)
                self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
            train_acc = float(np.mean(self.predict(x) == y))
            history.append({"epoch": step, "loss": float(loss), "accuracy": train_acc})
        return history


def metrics(y_true, y_pred, n_classes=3):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    accuracy = float(np.trace(matrix) / np.sum(matrix))
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i, i]
        precision = float(tp / matrix[:, i].sum()) if matrix[:, i].sum() else 0.0
        recall = float(tp / matrix[i, :].sum()) if matrix[i, :].sum() else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
    return {"accuracy": accuracy, "confusion_matrix": matrix.tolist(), "per_class": per_class}


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_data()
    train_df, test_df = stratified_split(df)
    x_train_raw = train_df[FEATURES].to_numpy(dtype=np.float64)
    x_test_raw = test_df[FEATURES].to_numpy(dtype=np.float64)
    y_train = train_df["target_id"].to_numpy(dtype=int)
    y_test = test_df["target_id"].to_numpy(dtype=int)
    mean, std = x_train_raw.mean(axis=0), x_train_raw.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train_raw - mean) / std
    x_test = (x_test_raw - mean) / std

    model = DenseNetwork()
    history = model.fit(x_train, y_train)
    train_metrics = metrics(y_train, model.predict(x_train))
    test_metrics = metrics(y_test, model.predict(x_test))
    pd.DataFrame(history).to_csv(RESULTS / "training_history.csv", index=False)
    pd.DataFrame(test_metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(RESULTS / "confusion_matrix.csv")
    report = {
        "dataset": {"total_samples": len(df), "train_samples": len(train_df), "test_samples": len(test_df), "features": FEATURES, "classes": CLASS_NAMES},
        "split": {"test_ratio": 0.2, "seed": SEED, "stratified": True},
        "architecture": {"layers": [4, 8, 8, 3], "hidden_activation": "ReLU", "output_activation": "Softmax"},
        "training": {"loss": "categorical cross-entropy", "optimizer": "Adam", "learning_rate": 0.01, "epochs": 500, "batch": "full batch"},
        "standardization": {"means": mean.tolist(), "stds": std.tolist()},
        "final_training_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"train_accuracy": train_metrics["accuracy"], "test_accuracy": test_metrics["accuracy"], "test_confusion_matrix": test_metrics["confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
