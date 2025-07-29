import cv2 as cv
import numpy
class ZoneMarker:
    def __init__(self):
        pass

    def get_bounding_box(self, img, consensus_mask):
        contours, _ = cv.findContours(consensus_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            x, y, w, h = cv.boundingRect(cont)
            if w * h > 500:  # skip very small regions
                # Draw rectangle on the image
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green box with thickness 2
        return img