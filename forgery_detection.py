import cv2
import numpy as np

def detect_differences(original_path, altered_path):
    # Read images
    original = cv2.imread(original_path)
    altered = cv2.imread(altered_path)

    # Resize to match dimensions
    min_width = min(original.shape[1], altered.shape[1])
    min_height = min(original.shape[0], altered.shape[0])
    original = cv2.resize(original, (min_width, min_height))
    altered = cv2.resize(altered, (min_width, min_height))

    # Convert to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    altered_gray = cv2.cvtColor(altered, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(original_gray, altered_gray)

    # Apply threshold to highlight differences
    _, threshold_diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Convert to color heatmap (Enhancing visibility)
    heatmap = cv2.applyColorMap(threshold_diff, cv2.COLORMAP_JET)

    # Increase weight of heatmap for better visibility
    overlaid = cv2.addWeighted(altered, 0.5, heatmap, 0.7, 0)

    return original, altered, overlaid
