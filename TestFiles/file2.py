import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import seaborn as sns
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

(train_dataset, test_dataset), info = tfds.load('fashion_mnist', split=['train', 'test'], with_info=True)
class_names = info.features['label'].names
print("Class labels: {}".format(info.features['label'].names))
print("Numbers of training examples: {}".format(info.splits['train'].num_examples))
print("Numbers of test examples: {}".format(info.splits['test'].num_examples))

def preprocessing(item):
  return (item["image"] / 255, item["label"])

train_dataset = train_dataset.map(preprocessing)
test_dataset = test_dataset.map(preprocessing)

batch_size = 64
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.shuffle(buffer_size = 1024, seed = 0)
train_dataset = train_dataset.batch(batch_size = batch_size)
train_dataset = train_dataset.prefetch(buffer_size = 1)

test_dataset = test_dataset.cache().batch(batch_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(64, activation='relu'),
  # tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

steps = 60000//64

history = model.fit(
    train_dataset,
    batch_size=64,
    epochs=30,
    verbose=1,
    steps_per_epoch=steps,
)

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(10000//64)) #10000 :test samples available
print('Accuracy on test dataset:', test_accuracy)

M_no = '002'

results_tmp = np.array([M_no, test_loss, test_accuracy]).reshape(1,-1)