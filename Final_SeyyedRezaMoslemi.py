import cv2
import numpy as np


#---------------- Section 5 & Bonus(Section 7) ----------------
def calculator(x1, y1, x2, y2, w, h):
    
    # Calculate the slope and intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Set the starting and ending points for the line
    x1 = w
    y1 = int(slope * x1 + intercept)

    x2 = int((h - intercept)/slope)
    y2 = h

    return x1,y1,x2,y2


def detector(image, prev_left_line=None, prev_right_line=None):
#---------------- Section 2 ----------------
    h, w = image.shape[:2]
    w //= 2
    h //= 2
    image = cv2.resize(image, (w, h))
    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    g_image = cv2.GaussianBlur(image_, (3, 3), 0)

    edges = cv2.Canny(g_image, threshold1=50, threshold2=300)


#---------------- Section 3 ----------------
    # Bounding image to five vector
    contour = np.array([[0, h], [w, h], [w//2, 6*h//10], [w//4, 6*h//10], [3*w//4, 6*h//10]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.int32([contour]), 255)
    masked_edges = cv2.bitwise_and(edges, mask)


#---------------- Section 4 ----------------
    # Perform Hough line detection using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=55,
        maxLineGap=50
    )


    #---------------- Section 4 & 5 & Bonus(Section 7) ----------------
    if lines is not None:
        left_lines = []  
        right_lines = []  

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)

            if slope < 0:  
                right_lines.append(line)
            elif slope > 0:  
                left_lines.append(line)

        if len(left_lines) > 0:
            left_line_avg = np.mean(left_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = left_line_avg[0]
            x1, y1, x2, y2 = calculator(x1, y1, x2, y2, w, 6*h//10)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            prev_left_line = (x1, y1, x2, y2)
        elif prev_left_line is not None:
            x1, y1, x2, y2 = prev_left_line
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if len(right_lines) > 0:
            right_line_avg = np.mean(right_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = right_line_avg[0]
            x1, y1, x2, y2 = calculator(x1, y1, x2, y2, 0, 6*h//10)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            prev_right_line = (x1, y1, x2, y2)
        elif prev_right_line is not None:
            x1, y1, x2, y2 = prev_right_line
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image, prev_left_line, prev_right_line


#---------------- Section 6 ----------------
def play_video(video):
    prev_left_line, prev_right_line = None, None
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        processed_frame, prev_left_line, prev_right_line = detector(frame, prev_left_line, prev_right_line)

        cv2.imshow('Processed Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



# main
video1 = cv2.VideoCapture('vid1.mp4')
play_video(video1)

video2 = cv2.VideoCapture('vid2.mp4')
play_video(video2)

image = cv2.imread('img1.jpg')
image,_,_ = detector(image, None, None)
cv2.imshow('ّFinal Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
