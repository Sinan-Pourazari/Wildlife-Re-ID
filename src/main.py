import cv2 as cv
from bg_tools import BackgroundSubtracktor
from roi_tools import ZoneMarker
import numpy as np
def main():
   
    video_path = r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\test_vid\Foxes\Fox_Nestler_BG (51).MOV"
    #animal_video_path = r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\test_vid\whole.mp4"
    animal_video_path = r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\test_vid\Foxes\Fox_Nestler_BG (2).MOV"
    #animal_video_path= r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\test_vid\door.mp4"
    bg_sub = BackgroundSubtracktor(video_path,20 ,60, alpha= 0.05, running_start=False)

    bg_sub.analyse_video(video_path=animal_video_path, frame_densnes=0, verbose=True)

if __name__ == "__main__":
    main()
