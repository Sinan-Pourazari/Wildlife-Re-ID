import cv2 as cv
import numpy as np
from video_tools import array_to_mp4
from tqdm import tqdm
import roi_tools as rt
class BackgroundSubtracktor:
    def __init__(self, background_video_path, segment_size, max_frames=None, running_start=True):
        # Initilize with the base backgrounds before doing anything else
        # Load background segments
        if not running_start:
            background_frames = self.__extract_frame_segments(background_video_path, segment_size, max_frames)
            self.num_backgrounds=(len(background_frames))
            print(f"Loaded {self.num_backgrounds} Initial background segments.")
            # Build background bases
            self.backgrounds = np.array([self.build_background_base(seg) for seg in background_frames])
        else:
            print('Collecting Initial Background segments during runtime')
        self.segment_size = segment_size
        self.confirmed_background_given = not running_start
        self.num_backgrounds = max_frames//segment_size
        #self.num_backgrounds=(len(background_frames))
        #self.backgrounds
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

        #background_uint8 = cv.dilate(background_uint8, (5, 5), 0)
        return background_uint8  # Still in LAB space

    def compare_images(self, new_image, base_background):
        new_lab = cv.cvtColor(new_image, cv.COLOR_BGR2LAB)
        diff = cv.absdiff(new_lab, base_background)

        l, a, b = cv.split(diff)
        _, l_ch = cv.threshold(l, 40, 255, cv.THRESH_BINARY)
        _, a_ch = cv.threshold(a, 50, 255, cv.THRESH_BINARY)
        _, b_ch = cv.threshold(b, 50, 255, cv.THRESH_BINARY)

        motion_mask = cv.bitwise_or(l_ch, a_ch)
        motion_mask = cv.bitwise_or(motion_mask, b_ch)
        #motion_mask = cv.dilate(motion_mask,(10,10))

        return motion_mask
    #TODO autotuning for thersholds
    #Todo make thesholds parameters
    def compare_images_rgb(self, new_image, base_background):
        diff = cv.absdiff(new_image, base_background)
        b, g, r = cv.split(diff)
        _, r_ch = cv.threshold(r, 80, 255, cv.THRESH_BINARY)
        _, g_ch = cv.threshold(g, 80, 255, cv.THRESH_BINARY)
        _, b_ch = cv.threshold(b, 80, 255, cv.THRESH_BINARY)

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

        avg_background /= frame_count

        background_uint8 = cv.convertScaleAbs(avg_background)
        background_uint8 = cv.GaussianBlur(background_uint8, (5, 5), 0)
        return background_uint8

    def __extract_frame_segments(self, video_path, segment_size=20, max_frames=None):
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

        frames_np = np.array(frames)
        frame_segments = np.array([
            frames_np[i:i + segment_size]
            for i in range(0, len(frames_np), segment_size)
            if len(frames_np[i:i + segment_size]) == segment_size
        ])

        return frame_segments

    def detect_motion_from_backgrounds(self, input_frame, backgrounds, weights, vote_threshold=0.95):
        """
        Compares an input frame against a list of background bases to detect motion.
        Returns: (consensus_mask, bounding_boxes)
        """
        motion_masks = np.array([self.compare_images(input_frame, bg) for bg in backgrounds])

        # Convert masks to binary (0/1)
        motion_masks_bin = np.array([(mask > 0).astype(np.uint8) for mask in motion_masks])
        
        #motion_masks_bin = []
        #for i in range(len(motion_masks)):
            #bin_mask = (motion_masks[i] > 0).astype(np.uint8) * weights[i]
            #motion_masks_bin.append(bin_mask) 

        #motion_masks_bin = np.array(motion_masks_bin)

        # Voting: sum up all pixel votes
        vote_map = np.tensordot(weights, motion_masks_bin, axes=(0, 0))
        #vote_map = np.sum(motion_masks_bin, axis=0)
        threshold = vote_threshold #* len(motion_masks)
        consensus_mask = (vote_map >= threshold).astype(np.uint8) * 255

        #perform Morphological closing to prevent boundingboxy fragmentation by conses noisse
        kernel = np.ones((5,5), np.uint8)
        consensus_mask = cv.morphologyEx(consensus_mask, cv.MORPH_CLOSE, kernel)


        contours, _ = cv.findContours(consensus_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #TODO make 250 a parameter 
        #TODO make it self learnable
        bounding_boxes = np.array([cv.boundingRect(cnt) for cnt in contours if cv.contourArea(cnt) > 150])


        return consensus_mask, bounding_boxes

    def init_background(self, frames):
        self.backgrounds = []  # Start with an empty list
        #background_frames = self.__extract_frame_segments(background_video_path, segment_size, max_frames)
        for frame in frames:
            base = self.build_background_base([frame])  # Process each segment
            self.backgrounds.append(base)           # Add to the list

        self.backgrounds = np.array(self.backgrounds)  # Convert list to NumPy array


    def analyse_video(self, video_path, frame_densnes, verbose= False):
        frames_np = self._load_video_frames(video_path)
        weights = self._init_weights()
        mask_history = []
        empty_sequence = []
        replace_index = 0
        consecutive_empty = False
        output_video = []
        frame_counter=0
        partial_frames_done =0

        tracker = rt.BBoxTracker()
        #TODO do not preload all frames
        if not self.confirmed_background_given:
            self._init_background_from_frames(frames_np)

        #itteration over the frames as they come in
        for frame in tqdm(frames_np, desc="Processing Frames", unit="frame"):
            consensus_mask, boxes = self.detect_motion_from_backgrounds(frame, self.backgrounds, weights)
            #smoothed_mask = self._apply_temporal_smoothing(mask_history, consensus_mask)
            boxes = self._extract_boxes(consensus_mask)

            empty_sequence, replace_index, consecutive_empty, weights = self._handle_background_update(
                frame, boxes, empty_sequence, replace_index, consecutive_empty, weights
            )
            #if motion is detected increment number of frames since last update if no motion is detected reset it
            if len(boxes) > 0:
                frame_counter += 1
            else:
                frame_counter = 0
            
            confirmed_tracks, _ = tracker.update(boxes)        # (list of dicts, list of dicts)
            print(confirmed_tracks)
            boxes_to_draw = [t["box"] + [t["tag"]] for t in confirmed_tracks]
            print(boxes_to_draw)
            annotated, bounding_boxes = self._draw_boxes(frame, boxes_to_draw)
            
            if frame_counter % 40 == 0 and frame_counter != 0:
                if verbose:
                    print("Partial Update triggered")
                
                self._partial_background_update(bounding_boxes,frame)
                partial_frames_done +=1
                if partial_frames_done >=40:
                    partial_frames_done=0
                    frame_counter=0

            cv.imshow("Live_view", annotated)
            cv.waitKey(1)
            output_video.append(annotated)

        cv.destroyAllWindows()
        array_to_mp4(np.array(output_video))

# --- Helper methods ---
    def _partial_background_update(self, bounding_boxes, frame ):
        """
        boundingboxes: array of arrays
        """
        newest_background = self.backgrounds[-1]
        for box in bounding_boxes:
            x, y, w, h = box
            frame[y:y+h, x:x+w] = newest_background[y:y+h, x:x+w]
                
        self._replace_background_frame(frame)

    def _replace_background_frame(self, new_frame, index=None):
        """
        Replaces a background slot with a new frame.
        If index is None, replaces the oldest (last) background.
        """
        if index is None:
            index = -1  # default: replace last background
        self.backgrounds[index] = self.build_background_base([new_frame])

    def _load_video_frames(self, video_path):
        """read video frames into numpy array"""
        cap = cv.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def _init_weights(self):
        """initialize weights for background voting"""
        weights = np.arange(1, self.num_backgrounds + 1)
        return weights / weights.sum()

    def _init_background_from_frames(self, frames_np):
        """set initial background bases if not given"""
        print(self.num_backgrounds)
        self.init_background(frames_np[:self.num_backgrounds])

    def _apply_temporal_smoothing(self, mask_history, mask, max_history=3):
        """smooth motion masks over recent frames"""
        mask_history.append(mask)
        if len(mask_history) > max_history:
            mask_history.pop(0)
        smoothed = np.mean(mask_history, axis=0).astype(np.uint8)
        return (smoothed > 50).astype(np.uint8) * 255

    def _extract_boxes(self, smoothed_mask):
        """find bounding boxes from smoothed mask"""
        contours, _ = cv.findContours(smoothed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return np.array([cv.boundingRect(cnt) for cnt in contours if cv.contourArea(cnt) > 300])

    def _handle_background_update(self, frame, boxes, empty_sequence, replace_index, consecutive_empty, weights):
        """update backgrounds based on empty frame sequences"""
        if len(boxes) == 0 and len(empty_sequence) < self.segment_size:
            if not consecutive_empty:
                empty_sequence = []
            empty_sequence.append(frame)
            consecutive_empty = True

            if len(empty_sequence) >= 1:
                new_base_segment = self.build_background_base(np.array(empty_sequence))
                self.backgrounds[replace_index] = new_base_segment
        else:
            if consecutive_empty:
                new_base_segment = self.build_background_base(np.array(empty_sequence))
                self.backgrounds[replace_index] = new_base_segment
                weights = self._init_weights()
                replace_index = (1 + replace_index) % self.num_backgrounds
                empty_sequence = []
            consecutive_empty = False
        return empty_sequence, replace_index, consecutive_empty, weights

    def _draw_boxes(self, frame, boxes):
        """draw bounding boxes on frame"""
        annotated = frame.copy()
        bounding_boxes = []
        for (x, y, w, h, tag) in boxes:
            cv.rectangle(annotated, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv.putText(annotated, tag, (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL,0.5, (0, 255, 0), 1, cv.LINE_AA)
            bounding_boxes.append([x,y,w,h])
        return annotated, bounding_boxes