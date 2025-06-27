import cv2

def blur_region(frame, x1, y1, x2, y2, ksize=51):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 30)
    frame[y1:y2, x1:x2] = blurred_roi
    return frame
