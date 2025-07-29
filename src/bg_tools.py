import cv2 as cv
import numpy as np
class BackgroundSubtracktor:
    def __init__(self):
        self.background = None

    def build_background_base_bw(self, image_series):
        avg_background = None
        frame_count = 0

        for image in image_series:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = image.astype("float32")

            if avg_background is None:
                avg_background = image
            else:
                avg_background += image

                frame_count += 1

        avg_background /= frame_count
        background_uint8 = cv.convertScaleAbs(avg_background)

        # Noise reduction and cleaning
        #background_uint8 = cv.GaussianBlur(background_uint8, (5, 5), 0)
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        #background_uint8 = cv.morphologyEx(background_uint8, cv.MORPH_OPEN, kernel)
        #background_uint8 = cv.morphologyEx(background_uint8, cv.MORPH_CLOSE, kernel)

        return background_uint8
    
    def build_background_base(self, image_series):
        avg_background = None
        frame_count = 0

        for image in image_series:
            lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
            lab_image = lab_image.astype("float32")

            if avg_background is None:
                avg_background = lab_image
            else:
                avg_background += lab_image

                frame_count += 1

        avg_background /= frame_count
        background_uint8 = cv.convertScaleAbs(avg_background)

        # Noise reduction and cleaning
       # background_uint8 = cv.GaussianBlur(background_uint8, (5, 5), 0)
       # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
       # background_uint8 = cv.morphologyEx(background_uint8, cv.MORPH_OPEN, kernel)
       # background_uint8 = cv.morphologyEx(background_uint8, cv.MORPH_CLOSE, kernel)

        return background_uint8  # Still in LAB space

    def compare_images(self, new_image, base_background):
        new_lab = cv.cvtColor(new_image, cv.COLOR_BGR2LAB)
        diff = cv.absdiff(new_lab, base_background)

        l, a, b = cv.split(diff)
        _, l_ch = cv.threshold(l, 1, 255, cv.THRESH_BINARY)
        _, a_ch = cv.threshold(a, 50, 255, cv.THRESH_BINARY)
        _, b_ch = cv.threshold(b, 50, 255, cv.THRESH_BINARY)

        motion_mask = cv.bitwise_or(l_ch, a_ch)
        motion_mask = cv.bitwise_or(motion_mask, b_ch)

        return motion_mask
    
    #TODO change use compare images in build background base instead of using averages
    def compare_images_bw(self, new_image, base_background):
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(new_image, base_background)
    # Threshold the difference to get motion areas
        _, motion_mask = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)
        #motion_mask = cv.bitwise_or(new_image,base_background)
        return motion_mask
    


    def compare_images_rgb(self, new_imgae, base_background):
            diff = cv.absdiff(new_imgae, base_background)
            
            b, g, r = cv.split(diff)
            _, r_ch = cv.threshold(r, 70, 255, cv.THRESH_BINARY)
            _, g_ch = cv.threshold(g, 70, 255, cv.THRESH_BINARY)
            _, b_ch = cv.threshold(b, 70, 255, cv.THRESH_BINARY)

            motion_mask = cv.bitwise_or(b_ch, g_ch)
            motion_mask = cv.bitwise_or(motion_mask, r_ch)




            return motion_mask


    def build_background_base_rgb(self, image_series):
        avg_background = None
        frame_count = 0

        for image in image_series:
            image = image.astype("float32")

            if avg_background is None:
                avg_background = image
            else:
                avg_background += image

                frame_count += 1

        # Compute the average (element-wise)
        avg_background /= frame_count

        # Convert back to uint8 for further OpenCV usage
        background_uint8 = cv.convertScaleAbs(avg_background)
        #Pixel Level Noise Reduction
        background_uint8 = cv.GaussianBlur(background_uint8, (5, 5), 0)
        # Clean noise from motion mask
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        #background_uint8 = cv.morphologyEx(background_uint8, cv.MORPH_OPEN, kernel)
        #background_uint8 = cv.morphologyEx(background_uint8, cv.MORPH_CLOSE, kernel)

        return background_uint8
    
    def detect_movement(self,image):
        self.compare_images(image)

        pass

    def extract_frame_segments(video_path, segment_size=20, max_frames=None):
        cap = cv.VideoCapture(video_path)
        frames = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            count += 1

            if max_frames is not None and count >= max_frames:
                break

        cap.release()

        # Now group frames into segments of segment_size
        frame_segments = [
            frames[i:i + segment_size]
            for i in range(0, len(frames), segment_size)
            if len(frames[i:i + segment_size]) == segment_size  # only full segments
        ]

        return frame_segments


