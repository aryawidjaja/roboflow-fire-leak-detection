import cv2
import base64
import requests
import os
import time

# Set your API key and model ID
API_KEY = "your_roboflow_api_key"
MODEL_ID = "fire-smoke-leak-detection/1"
API_URL = f"http://localhost:9001/{MODEL_ID}?api_key={API_KEY}"

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# Create directory to store captured images
if not os.path.exists('captured'):
    os.makedirs('captured')

# Define colors for each class
colors = {
    "fire": (0, 0, 255),  # Red
    "smoke": (255, 0, 0),  # Blue
    "leak": (0, 255, 0)  # Green
}

def crop_and_resize(image, size):
    h, w, _ = image.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped_image = image[top:top+min_dim, left:left+min_dim]
    resized_image = cv2.resize(cropped_image, (size, size))
    return resized_image, top, left, min_dim

# Run inference
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Crop and resize frame to 640x640
    frame_resized, top, left, min_dim = crop_and_resize(frame, 640)

    # Encode the frame to base64
    _, img_encoded = cv2.imencode('.jpg', frame_resized)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Send the frame to the inference API
    response = requests.post(API_URL, data=img_base64)
    result = response.json()

    # Draw bounding boxes on the frame
    predictions = result['predictions']
    if predictions:
        for prediction in predictions:
            x0 = int((prediction['x'] - prediction['width'] / 2) * min_dim / 640 + left)
            y0 = int((prediction['y'] - prediction['height'] / 2) * min_dim / 640 + top)
            x1 = int((prediction['x'] + prediction['width'] / 2) * min_dim / 640 + left)
            y1 = int((prediction['y'] + prediction['height'] / 2) * min_dim / 640 + top)
            confidence = prediction['confidence']
            class_name = prediction['class']

            # Draw rectangle
            color = colors.get(class_name, (0, 255, 0))  # Default to green if class not found
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            # Put label
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        
        # Save the frame with bounding boxes
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f'captured/hazard_detected_{timestamp}.jpg', frame)
    else:
        cv2.putText(frame, "No hazard detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw ROI on the original frame
    cv2.rectangle(frame, (left, top), (left + min_dim, top + min_dim), (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
