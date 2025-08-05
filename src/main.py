import cv2 as cv
from bg_tools import BackgroundSubtracktor
from roi_tools import ZoneMarker
import numpy as np
def main():
   
    video_path = r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\test_vid\background.mp4"
    animal_video_path = r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\test_vid\whole.mp4"
    
    bg_sub = BackgroundSubtracktor(video_path,10,200)

    bg_sub.analyse_video(video_path=animal_video_path, frame_densnes=0)
    # Show results
    #cv.imshow("Animal Image", animal_image)
    #cv.imshow("Consensus Mask", consensus_mask)
    #cv.imshow("Annotated", annotated)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

if __name__ == "__main__":
    main()
