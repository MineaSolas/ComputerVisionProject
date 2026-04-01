import json
import numpy as np
import cv2

# 1. Load the Label Studio JSON export
# (Update the filename to match your exported file)
json_file = r'roi_masks\top_view_roi_mask.json'

with open(json_file, 'r') as f:
    data = json.load(f)[0] # Get the first (and only) annotated image

# 2. Extract the annotation data
# Navigate through the JSON structure to find the polygon
result = data['annotations'][0]['result'][0]
original_width = result['original_width']
original_height = result['original_height']
points_percentage = result['value']['points']

# 3. Convert percentage coordinates to absolute pixel coordinates
points_absolute = []
for x_pct, y_pct in points_percentage:
    # Convert from percentage (e.g., 50.5) to actual pixel (e.g., 960)
    x = int((x_pct / 100.0) * original_width)
    y = int((y_pct / 100.0) * original_height)
    points_absolute.append([x, y])

# Convert the list to a NumPy array for OpenCV
points_array = np.array([points_absolute], dtype=np.int32)

# 4. Create the blank mask and draw the polygon
# Create a black background (zeros)
mask = np.zeros((original_height, original_width), dtype=np.uint8)

# Fill the polygon with white (255)
cv2.fillPoly(mask, points_array, 255)

# 5. Save the final mask
cv2.imwrite('top_view_roi_mask.png', mask)