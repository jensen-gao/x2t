import cv2
import pickle
import os
import imageio
import json
import numpy as np
from datetime import datetime
from PIL import Image
from pygifsicle import optimize

from emnist import extract_training_samples, extract_test_samples


def get_space_data(split='train', space_index=7):
    """
    Get images and labels for representing the 'SPACE' character.
    """
    if split == 'train':
        images, labels = extract_training_samples('mnist')
    else:
        images, labels = extract_test_samples('mnist')

    space_indices = np.where(labels == space_index)[0]
    space_images = images[space_indices]
    space_labels = np.zeros(len(space_images), dtype=int)
    return space_images, space_labels


def uji_to_image(strokes, noise=None, margin=0.2, thickness=1, output_size=28):
    """
    Converts a list of pen strokes for a drawing to an numpy array representing the image of the drawing
    """

    # calculates max/min coordinates in image
    max_x, max_y = np.max(np.concatenate(strokes, axis=0), axis=0)
    min_x, min_y = np.min(np.concatenate(strokes, axis=0), axis=0)

    # calculates dimension of the image (if no margin)
    x_len = max_x - min_x
    y_len = max_y - min_y

    # add noise to the velocities in the strokes
    if noise is not None:
        # dimension of the image if square
        img_len = max(x_len, y_len)
        noisy_strokes = []

        for stroke, n in zip(strokes, noise):
            # noise proportional to image dimension
            noisy_velocities = np.concatenate((stroke[0:1], stroke[1:] - stroke[:-1]), axis=0) + n * img_len
            noisy_strokes.append(np.round(np.cumsum(noisy_velocities, axis=0)).astype(int))
        strokes = noisy_strokes

        # recalculate for noisy strokes
        max_x, max_y = np.max(np.concatenate(strokes, axis=0), axis=0)
        min_x, min_y = np.min(np.concatenate(strokes, axis=0), axis=0)

        x_len = max_x - min_x
        y_len = max_y - min_y

    # dimension of the image if square, with margin
    img_len = int(round(max(x_len, y_len) * (1 + margin)))

    # calculate coordinate to center positions around
    start_coord = np.round([(min_x + max_x - img_len) / 2, (min_y + max_y - img_len) / 2]).astype(int)
    image = np.zeros((img_len, img_len))

    # thickness of strokes scaled to size of image
    thickness = np.round(thickness * img_len / output_size).astype(int)

    # draw lines between positions in a stroke
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            cv2.line(image, tuple(stroke[i] - start_coord), tuple(stroke[i + 1] - start_coord),
                     (255, 255, 255), thickness=thickness)

    # reshape image to desired shape, normalize to 0-1
    return cv2.resize(image, (output_size, output_size)) / 255


def test_image_noise(step_brown_std, noise_std, drift_std, user_index=0, steps_per_sample=20, samples_per_period=10,
                     trials=5):
    """
    For visualizing effect of noise drift on drawings over time.
    """

    time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    save_path = os.path.join('./noisy_uji', time)
    os.makedirs(save_path)
    params = locals()

    del params['time']
    del params['save_path']

    with open(os.path.join(save_path, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    with open('./data/online_handwriting.pkl', 'rb') as f:
        data = pickle.load(f)
    for key in data.keys():
        data[key] = data[key][2 * user_index:2 * (user_index + 1)]
    data['space'] = data.pop(' ')

    brown_std = np.sqrt(steps_per_sample) * step_brown_std
    for key, value in data.items():
        strokes = value[0]

        char_path = os.path.join(save_path, key)
        os.makedirs(char_path)

        original = Image.fromarray(np.uint8(cv2.resize(uji_to_image(strokes) * 255, (100, 100))), 'L')
        original.save(os.path.join(char_path, 'original.png'))

        for trial in range(trials):
            drift = np.random.normal(scale=drift_std, size=2)
            brown = 0
            trial_path = os.path.join(char_path, 'drift_' + str(trial))
            os.makedirs(trial_path)
            frames = []
            for i in range(samples_per_period):
                brown += np.random.normal(scale=brown_std, size=2)
                noise = [np.random.normal(scale=noise_std, size=stroke.shape) + brown + drift for stroke in strokes]
                image = Image.fromarray(np.uint8(cv2.resize((uji_to_image(strokes, noise) * 255), (100, 100))), 'L')
                frame_path = os.path.join(trial_path, 'sample_' + str(i) + '.png')
                image.save(frame_path)
                frames.append(imageio.imread(frame_path))

            gif_path = os.path.join(trial_path, 'progression.gif')
            imageio.mimsave(gif_path, frames, duration=0.2)
            optimize(gif_path)
