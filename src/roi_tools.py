import cv2 as cv
import numpy as np
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

class BBoxTracker:
    def __init__(self, min_area:float=0.4, alpha:float=0.4, max_misses:int=10, min_age:int=35):
        #contains all active tracks (list of dicts)
        self.tracks = []
        self.min_area=min_area
        self.alpha = alpha
        self.next_id=0
        self.max_misses=max_misses
        self.min_age = min_age

    def update(self, boxes):
        #d[2] * d[3] = w*h-> only  keep boxes that are large enough
        qualified_boxes = [list(map(int, d)) for d in boxes if d[2] * d[3] >= self.min_area]
        
        #get all acitve tracks
        prev_boxes =[t["box"] for t in self.tracks]
        
        matches, unmatched_prev, unmatched_curr = self.greedy_matching(prev_boxes, qualified_boxes)

        #update known tracks
        for prev_index, detec_index in matches:
            track = self.tracks[prev_index]
            x,y,w,h = track["box"]
            nx,ny,nw,nh = qualified_boxes[detec_index]
            #update track position
            track["box"] =[int(self.alpha *x + (1-self.alpha)*nx) -10,
                           int(self.alpha *y + (1-self.alpha)*ny) -10,
                           int(self.alpha *w + (1-self.alpha)*nw) +20,
                           int(self.alpha *h + (1-self.alpha)*nh)+ 20
                           ]
            track["age"] +=1
            track["misses"]=0
            if(track["age"]>= self.min_age):
                track["tag"]= f"Confirmed Track"
        #unmatched previous: known tracks that have been lost, possible occlussion
        for prev_index in unmatched_prev:
            self.tracks[prev_index]["misses"] +=1
        
        #unmatched detections: create new tentative tracks
        for detec_index in unmatched_curr:
            
            self.tracks.append(
                {
                "id":self.next_id,
                "box":qualified_boxes[detec_index],
                "age": 1,
                "misses":0,
                "tag": f"Unconfirmed Track {self.next_id}"
                
                }
            )
            self.next_id +=1
        #remove all tracks that have been lost
        self.tracks = [t for t in self.tracks if t["misses"] <= self.max_misses]

        # confirmed only (temporal persistence at box level) 
        confirmed = [t for t in self.tracks if t["age"] >= self.min_age and t["misses"] == 0]

        # get all unconfiremd tracks
        unconfirmed = [t for t in self.tracks if t["age"] <= self.min_age and t["misses"] == 0]

        return confirmed, unconfirmed, self.tracks

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # convert to bottom-right coordinates
        ax2 = x1 + w1
        ay2 = y1 + h1
        bx2 = x2 + w2
        by2 = y2 + h2

        overlap_area = self._calc_area_of_overlap(x1, y1, ax2, ay2, x2, y2, bx2, by2)
        union_area = self._calc_union_area(w1, h1, w2, h2, overlap_area)

        return overlap_area / (union_area + 1e-6)  # add small epsilon to avoid /0
    
    def find_best_match(self, box_frame_one: np.ndarray, boxes_frame_two: np.ndarray):
        best_iou = None
        best_match_id = None
        for curr_id, next_box in enumerate(boxes_frame_two):
            iou = self.iou(box_frame_one, next_box)
            if best_match_id is None:
                best_match_id = curr_id
                best_iou = iou
            elif best_iou < iou:
                best_match_id = curr_id
                best_iou = iou
        
        return boxes_frame_two[best_match_id]

    def find_best_match_across_all(self, boxes_frame_one: np.ndarray, boxes_frame_two: np.ndarray):
        best_iou = None
        best_match_id = None
        best_match_id_2= None
        for curr_id, next_box in enumerate(boxes_frame_one):
                for curr_id2,next_box2 in enumerate(boxes_frame_two):
                    iou = self.iou(next_box2, next_box)
                    if best_match_id is None:
                        best_match_id = curr_id
                        best_match_id_2 = curr_id2
                        best_iou = iou
                    elif best_iou < iou:
                        best_match_id = curr_id
                        best_match_id_2 = curr_id2
                        best_iou = iou
        
        return boxes_frame_two[best_match_id_2]
    
    def greedy_matching(self, boxes_frame_one: np.ndarray, boxes_frame_two: np.ndarray, iou_thresh: float = 0.3):
        unmatched_prev=[]
        unmatched_curr=[]
        matches=[]
        used_boxes_frame_1 = []
        used_boxes_frame_2 = []
        all_matches = []
        
        #calculate all ious over all possible configurations
        for curr_id, next_box in enumerate(boxes_frame_one):
                for curr_id2,next_box2 in enumerate(boxes_frame_two):
                    iou = self.iou(next_box, next_box2)
                    if iou >= iou_thresh:
                        all_matches.append([curr_id,curr_id2,iou])
                

        #greedyly selsect mathces based on  highest iou where non of each box ids is already taken
        pairs_sorted = sorted(all_matches, key=lambda x: x[2], reverse=True)
        for triple in pairs_sorted:
            
            if triple[0] not in used_boxes_frame_1 and triple[1] not in used_boxes_frame_2:
                matches.append((triple[0], triple[1]))
                used_boxes_frame_1.append(triple[0])
                used_boxes_frame_2.append(triple[1])

        #all boxes that werent used are ssorted into the according lsits by id
        for i in range(len(boxes_frame_one)):
            if i not in used_boxes_frame_1:
                unmatched_prev.append(i)

        for i in range(len(boxes_frame_two)):
            if i not in used_boxes_frame_2:
                unmatched_curr.append(i)

        return matches, unmatched_prev, unmatched_curr
            

    # ---- Helper methods ----- #
    
    def _calc_area_of_overlap(self, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        """
        Calculate the overlapping area between two rectangles.

        Parameters:
        ax1, ay1 : int - Top-left corner of box A (x, y)
        ax2, ay2 : int - Bottom-right corner of box A (x, y)
        bx1, by1 : int - Top-left corner of box B (x, y)
        bx2, by2 : int - Bottom-right corner of box B (x, y)

        Returns:
        int - Area of the overlap in pixels (0 if no overlap)
        """
        
        ix1 = max(ax1, bx1)   # left edge
        iy1 = max(ay1, by1)   # top edge
        ix2 = min(ax2, bx2)   # right edge
        iy2 = min(ay2, by2)   # bottom edge

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        return iw * ih


    def _calc_union_area(self, w1, h1, w2, h2, overlap_area):
        """
        Calculates the area of both boxes.

        Parameters:
        w1, w2 : width of each box.
        h1, h2 : height of each box.
        overlap area : overlap of both boxes
        """
        area_a = w1 * h1
        area_b = w2 * h2
        return area_a + area_b - overlap_area


