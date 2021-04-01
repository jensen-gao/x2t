import os
import json
import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples
from datetime import datetime
import argparse
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import random
from utils import get_space_data

# For training the default EMNIST classifier


tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', '-l', type=str, default=None,
                    help='Path to load weights from.')
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=32,
                    help='Training and validation batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4,
                    help='Learning rate.')

args = parser.parse_args()

train_images, train_labels = extract_training_samples('letters')

train_space_images, train_space_labels = get_space_data('train')

train_images = np.concatenate((train_images, train_space_images), axis=0)
train_labels = np.concatenate((train_labels, train_space_labels), axis=0)

valid_data = list(zip(*extract_test_samples('letters')))

state = random.getstate()
random.seed(123456)
random.shuffle(valid_data)
random.setstate(state)

valid_images, valid_labels = zip(*valid_data)

valid_space_images, valid_space_labels = get_space_data('test')

valid_images = np.concatenate((valid_images, valid_space_images), axis=0)
valid_labels = np.concatenate((valid_labels, valid_space_labels), axis=0)

train_images, valid_images = np.expand_dims(train_images, -1) / 255, np.expand_dims(valid_images, -1) / 255

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))

train_dataset = train_dataset.shuffle(buffer_size=100000).batch(args.batch_size)
valid_dataset = valid_dataset.batch(args.batch_size)

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1), name='c1'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (5, 5), activation='relu', name='c2'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(27, name='q_values'))

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])

if args.load_path is not None:
    model.load_weights(args.load_path, by_name=True)

time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
model_path = 'offline/models/'
save_dir = os.path.join(model_path, time)
checkpoint_dir = os.path.join(save_dir, 'checkpoints')
os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

with open(os.path.join(save_dir, 'params.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

history = model.fit(train_dataset, epochs=args.epochs, callbacks=[checkpoint_callback], validation_data=valid_dataset)
model.save(os.path.join(save_dir, 'model'))

with open(os.path.join(save_dir, 'train_hist.txt'), 'w') as f:
    print(history.history, file=f)
