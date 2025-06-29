"""
Utility functions for digit classification
"""

import matplotlib.pyplot as plt
from sklearn import metrics


def flatten_images(images):
    """
    Flatten images from 2D arrays to 1D arrays.
    
    Args:
        images: Array of 2D images
        
    Returns:
        Flattened array suitable for ML algorithms
    """
    n_samples = len(images)
    return images.reshape((n_samples, -1))


def visualize_sample_images(images, targets, title_prefix="Training", n_samples=4):
    """
    Visualize sample images with their labels.
    
    Args:
        images: Array of images to display
        targets: Corresponding labels
        title_prefix: Prefix for image titles
        n_samples: Number of samples to display
        
    Returns:
        matplotlib figure and axes
    """
    _, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=(10, 3))
    for ax, image, label in zip(axes, images[:n_samples], targets[:n_samples]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{title_prefix}: {label}")
    return axes


def visualize_predictions(X_test, predictions, n_samples=4):
    """
    Visualize test samples with their predictions.
    
    Args:
        X_test: Test data (flattened)
        predictions: Model predictions
        n_samples: Number of samples to display
        
    Returns:
        matplotlib axes
    """
    _, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test[:n_samples], predictions[:n_samples]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    return axes


def print_classification_metrics(classifier, y_test, predictions):
    """
    Print comprehensive classification metrics.
    
    Args:
        classifier: Trained classifier object
        y_test: True test labels
        predictions: Model predictions
    """
    print(
        f"Classification report for classifier {classifier}:\n"
        f"{metrics.classification_report(y_test, predictions)}\n"
    )


def display_confusion_matrix(y_test, predictions, title="Confusion Matrix"):
    """
    Display confusion matrix with visualization.
    
    Args:
        y_test: True test labels
        predictions: Model predictions
        title: Title for the confusion matrix
        
    Returns:
        ConfusionMatrixDisplay object
    """
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle(title)
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp


def confusion_matrix_to_classification_report(confusion_matrix):
    """
    Convert confusion matrix back to classification report format.
    
    Args:
        confusion_matrix: 2D array representing confusion matrix
        
    Returns:
        Tuple of (y_true, y_pred) lists
    """
    y_true = []
    y_pred = []
    
    for gt in range(len(confusion_matrix)):
        for pred in range(len(confusion_matrix)):
            y_true += [gt] * confusion_matrix[gt][pred]
            y_pred += [pred] * confusion_matrix[gt][pred]
    
    return y_true, y_pred
