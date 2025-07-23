import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from tqdm import tqdm

from cycle_schedule import CycleScheduling

seed = 42

housing = fetch_california_housing()
X = housing['data']
y = housing['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

# Standardization, centering the data with mean 0 and std 1
# Read: https://en.wikipedia.org/wiki/Standard_score
scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defining the L2 Regularizer. alpha * \sum {w[i]}
# where w[i] is the ith weight.
l2_regularizer = keras.regularizers.l2(0.05)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "elu", kernel_initializer = "he_normal", kernel_regularizer = l2_regularizer),
    keras.layers.Dense(1, kernel_regularizer = l2_regularizer)
])

def random_batch(X, y, batch_size = 32):
    indices = np.random.randint(len(X), size = batch_size)
    return X[indices], y[indices]

def print_status_bar(iteration, total, loss, metrics = None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else '\n'
    print("\r{}/{} - ".format(iteration, total) + metrics, end = end)

def huber_loss(threshold = 1.0):
    # Huber loss, (e < threshold ? (|e| \cdot threshold - e**2 / 2))
    # Read: https://en.wikipedia.org/wiki/Huber_loss
    def huber(y_true, y_pred):
        e = y_true - y_pred
        de = tf.abs(e) < threshold
        se = tf.square(e) / 2
        le = threshold * tf.abs(e) - threshold**2 / 2
        return tf.where(de, se, le)
    return huber

class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber = huber_loss(threshold)
        self.total = self.add_weight(name = "total", initializer = 'zeros')
        self.count = self.add_weight(name = "count", initializer = 'zeros')
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        metric = self.huber(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config() # retains the base configurations
        return {**base_config, "threshold": self.threshold}

epochs = 22
batch_size = 54
steps = len(X_train) // batch_size

scheduler = CycleScheduling(max_lr = 0.01, total_steps = steps)
optimizer = keras.optimizers.Nadam(learning_rate = 0.01)
loss_function = keras.losses.MeanSquaredError()
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError(), HuberMetric()]

for epoch in range(1, epochs + 1):
    print("Epoch {}/{}".format(epoch, epochs))
    progress_bar = tqdm(range(1, steps + 1), desc = "Training", leave = False)
    for step in progress_bar:
        X_batch, y_batch = random_batch(X_train_scaled, y_train)

        lr = scheduler.get_lr()
        lr = float(lr)
        optimizer.learning_rate.assign(lr)

        # Backpropagation
        with tf.GradientTape() as tape:
            predictions = model(X_batch, training = True)
            main_loss = tf.reduce_mean(loss_function(y_batch, predictions))
            loss = tf.add_n([main_loss] + model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mean_loss(loss)

        for metric in metrics:
            metric(y_batch, predictions)

        progress_bar.set_postfix({
            "lr": f"{lr: .5f}",
            "loss": f"{mean_loss.result().numpy(): .4f}",
            "mean_absolute_error": f"{metrics[0].result().numpy(): .4f}",
            "huber_loss": f"{metrics[1].result().numpy(): .4f}"
        })
        
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_state()
