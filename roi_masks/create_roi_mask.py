import json
import numpy as np
import cv2
from pathlib import Path

dir_path = Path(r"C:\Users\sebas\PycharmProjects\Debris_Estimation\ComputerVisionProject\roi_masks")

# Iterate through all .json files in the specified directory
for json_file in dir_path.glob("*.json"):

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Navigate through the JSON structure to find the polygon
    result = data['annotations'][0]['result'][0]
    original_width = result['original_width']
    original_height = result['original_height']
    points_percentage = result['value']['points']

    # Convert percentage coordinates to absolute pixel coordinates
    points_absolute = []
    for x_pct, y_pct in points_percentage:
        x = int((x_pct / 100.0) * original_width)
        y = int((y_pct / 100.0) * original_height)
        points_absolute.append([x, y])

    # Convert the list to a NumPy array for OpenCV
    points_array = np.array([points_absolute], dtype=np.int32)

    # Create the blank mask and draw the polygon
    mask = np.zeros((original_height, original_width), dtype=np.uint8)

    # Fill the polygon with white (255)
    cv2.fillPoly(mask, points_array, 255)

    cv2.imwrite(f"{json_file.stem}.png", mask)