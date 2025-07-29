import cv2 as cv
from bg_tools import BackgroundSubtracktor
from roi_tools import ZoneMarker
import numpy as np
def main():
    max_frames = 200
    segment_size = 15
    num_base_backgrounds = max_frames // segment_size
    video_path = r"C:\Users\sinan\Desktop\Projekte\Wildlife-Re-ID\src\test_vid\background.mp4"
    animal_image_path = r"C:\Users\sinan\Desktop\Projekte\Wildlife-Re-ID\src\test_vid\fox.png"

    # Load background frame segments
    background_frames = BackgroundSubtracktor.extract_frame_segments(video_path, segment_size=segment_size, max_frames=max_frames)
    print(f"Loaded {len(background_frames)} background segments.")

    # Load animal image
    animal_image = cv.imread(animal_image_path)
    #animal_image = cv.cvtColor(animal_image, cv.COLOR_BGR2GRAY)
    if animal_image is None:
        print("Error loading animal image.")
        return

    if animal_image.shape != background_frames[0][0].shape:
        print("Resizing animal image to match background resolution.")
        h, w = background_frames[0][0].shape[:2]
        animal_image = cv.resize(animal_image, (w, h))

    # Build background models
    bg_sub = BackgroundSubtracktor()
    backgrounds = []
    # TODO change this so it takes the last few frames for the background as an initial base. (later replace by more recent frames)
    for segment in background_frames[:num_base_backgrounds]:
        bg = bg_sub.build_background_base_rgb(segment)
        backgrounds.append(bg)

    # Compare and collect motion masks
    motion_masks = []
    for bg in backgrounds:
        mask = bg_sub.compare_images_rgb(animal_image, bg)
        motion_masks.append(mask)

    # Convert all masks to 0/1 binary format
    motion_masks_bin = [(mask > 0).astype(np.uint8) for mask in motion_masks]

    # Sum up motion votes per pixel
    vote_map = np.sum(motion_masks_bin, axis=0)

    # Set majority threshold (e.g., motion in at least 2 out of 3 masks)
    #threshold = len(motion_masks) // 2 + 1
    threshold = int(0.80 * len(motion_masks))

    # Create final consensus mask
    consensus_mask = (vote_map >= threshold).astype(np.uint8) * 255
    #consensus_mask = 255 - consensus_mask 
    # Show results
    cv.imshow("Animal Image", animal_image)
    zone = ZoneMarker()
    img = zone.get_bounding_box(animal_image,consensus_mask)
    cv.imshow("ROI", img)
    #for i, mask in numerate(motion_masks):
        #cv.imshow(f"Motion Mask {i+1}", mask)
    #for i, bg in enumerate(backgrounds):
        #cv.imshow(f"Background Model {i+1}", bg)
    cv.imshow("Consensus", consensus_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save to current directory
    cv.imwrite("animal_image.jpg", animal_image)
    cv.imwrite("motion_mask.jpg", motion_masks[0])
    cv.imwrite("background_model.jpg", backgrounds[0])
    cv.imwrite("consensus.jpg", consensus_mask)

if __name__ == "__main__":
    main()