---

# Line Detection with OpenCV

This Python script uses OpenCV to perform lane detection in images and videos. It utilizes edge detection, Hough line transformation, and averaging to identify and draw the left and right lines on the road.

## Features

- **Line Detection**: Detects left and right lanes in images and videos.
- **Dynamic Visualization**: Displays processed frames in real-time with OpenCV.
- **Configurable Parameters**: Easily adjust parameters such as Canny edge detection thresholds, Hough line transformation parameters, and more.

## Usage

1. **Install Dependencies**: Make sure you have the required dependencies installed. You can install them using the following:

    ```bash
    pip install opencv-python numpy
    ```

2. **Run the Script**: Execute the script by running the following command:

    ```bash
    python Final_SeyyedRezaMoslemi.py
    ```

3. **Interact with Video**: Press 'q' to exit the video display window.

## Examples

### Video Processing

```python
# main
video1 = cv2.VideoCapture('vid1.mp4')
play_video(video1)

video2 = cv2.VideoCapture('vid2.mp4')
play_video(video2)
```

### Image Processing

```python
image = cv2.imread('img1.jpg')
image, _, _ = detector(image, None, None)
cv2.imshow('Final Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Notes

- The script employs a dynamic visualization approach, allowing real-time feedback during video processing.
- Lane detection is achieved by identifying edges, creating a mask, and using Hough line transformation.

Feel free to contribute, report issues, or suggest improvements!

---
