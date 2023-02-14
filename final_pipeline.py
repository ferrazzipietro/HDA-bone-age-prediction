from functools import reduce, cache

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def prepare_resnet(image):
    size = 244
    img = np.reshape(image, (299, 299, 1))
    img = tf.image.resize(img, (size, size))
    img = tf.concat((img, img, img), axis=2)
    img = tf.cast(img, tf.float32)
    tf.keras.applications.resnet.preprocess_input(image)

    dataset = tf.data.Dataset.from_tensor_slices((([img],), [0]))
    dataset = dataset.batch(batch_size=1)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


def process_features(features):
    return np.max(features, axis=(0, 1))


def create_single_input(image, extra_features):
    try:
        extra_features = float(extra_features)
    except TypeError:
        extra_features = list(map(float, extra_features))
    dataset = tf.data.Dataset.from_tensor_slices((([tf.convert_to_tensor(image / 255)], [extra_features]), [0]))

    dataset = dataset.batch(batch_size=1)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


@cache
def load_model(path):
    return tf.keras.models.load_model(path)


def find_valleys(histogram, peaks):
    valleys = []
    previous = peaks[0][0]
    for pos, _ in peaks[1:]:
        sublist = histogram[previous:pos]
        valley = sublist.argmin() + previous
        valleys.append((valley, histogram[valley]))
        previous = pos
    return valleys


def find_peaks(arr, width=3, min_amount=200):
    peaks = []
    for i in range(1, len(arr)):
        if arr[i] > min_amount and \
                all(arr[i] >= x for x in arr[max(0, i - width // 2):min(len(arr), i + 1 + width // 2)]):
            peaks.append((i, arr[i]))
    return peaks


def remove_noise(image, ellipse_size=5, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ellipse_size, ellipse_size))
    denoised = image.copy()
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel, iterations=iterations)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return denoised


def remove_lines(image, line_width=20):
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_width))
    removed = cv2.morphologyEx(padded, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
    removed = cv2.morphologyEx(removed, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
    removed = removed[1:-1, 1:-1]
    return removed


def get_range_mask(image, lower, upper):
    _, mask = cv2.threshold(image, upper, 255, cv2.THRESH_TOZERO_INV)
    _, mask = cv2.threshold(mask, lower, 255, cv2.THRESH_BINARY)
    return mask


def get_components(image):
    hist, bin = np.histogram(image.ravel(), 256, [0, 255])
    peaks = find_peaks(hist, 9)
    valleys = [i for i, _ in find_valleys(hist, peaks)]
    iter_valleys = zip([0, *valleys], [*valleys, 255])
    components = [get_range_mask(image, lower, upper) for lower, upper in iter_valleys]
    return components


def is_background_component(component):
    binary = component == 255
    width = component.shape[1]
    start = int(width * 0.3)
    end = int(width * 0.7)
    sum_center = np.sum(binary[:, start:end])
    sum_sides = np.sum(binary[:, :start]) + np.sum(binary[:, end:])
    horizontal_centrality = sum_center / (sum_center + sum_sides)

    height = component.shape[0]
    start = int(height * 0.2)
    end = int(height * 0.8)
    sum_center = np.sum(binary[start:end])
    sum_sides = np.sum(binary[:start]) + np.sum(binary[end:])
    vertical_centrality = sum_center / (sum_center + sum_sides)

    return horizontal_centrality < 0.6 or vertical_centrality < 0.6 or horizontal_centrality + vertical_centrality < 1


def get_hand_components(image):
    components = get_components(image)
    hand_components = [component for component in components if not is_background_component(component)]
    if len(hand_components) > 0:
        return reduce(cv2.bitwise_or, hand_components)
    copy = image.copy()
    copy[image != 0] = 255
    return copy


def get_hand(image):
    mask = get_hand_components(image)

    mask = remove_noise(mask)
    mask = remove_lines(mask, line_width=15)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = mask == 255
    masked = image.copy()
    masked[~mask] = 0

    y_nonzero, x_nonzero = np.nonzero(masked)
    cropped = masked[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    minval = np.min(cropped[np.nonzero(cropped)])
    maxval = np.max(cropped[np.nonzero(cropped)])
    rescaled = np.interp(cropped, (minval, maxval), (0, 255))
    rescaled = rescaled.astype(np.uint8)

    return rescaled


def plot_side_by_side(images, titles=None, max_per_column=4, labels=True):
    titles = titles or [''] * len(images)
    if len(images) <= max_per_column:
        for i, (image, title) in enumerate(zip(images, titles), 1):
            plt.subplot(1, len(images), i)
            plt.imshow(image, cmap='gray')
            plt.title(title)
            if not labels:
                plt.xticks([])
                plt.yticks([])
    else:
        shape = (np.ceil(len(images) / max_per_column), max_per_column)
        for i in range(int(shape[0])):
            for j in range(int(shape[1])):
                index = i * max_per_column + j
                if index >= len(images):
                    break
                plt.subplot(int(shape[0]), int(shape[1]), index + 1)
                plt.imshow(images[index], cmap='gray')
                plt.title(titles[index])
                if not labels:
                    plt.xticks([])
                    plt.yticks([])


def reshape(image, size, pad=True):
    scale_factor = max(image.shape) / size
    new_size = (int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor))
    resized = cv2.resize(image, new_size)
    top = int(np.ceil((size - new_size[1]) / 2))
    bottom = int(np.floor((size - new_size[1]) / 2))
    left = int(np.ceil((size - new_size[0]) / 2))
    right = int(np.floor((size - new_size[0]) / 2))
    if pad:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
    return resized


def process_image(src):
    try:
        image = cv2.imread(src, 0) if type(src) == str else src
        hand = get_hand(image)
        resized = reshape(hand, 299)
        return resized
    except ValueError:
        print(f'Error processing image {src}')
        return None
