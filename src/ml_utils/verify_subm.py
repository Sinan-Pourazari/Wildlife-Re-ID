import pandas as pd
import sys

def verify_submission(submission_path, master_metadata_path):
    print(f"--- AnimalCLEF ID Verification ---")
    
    # 1. Load the files
    try:
        sub_df = pd.read_csv(submission_path)
        meta_df = pd.read_csv(master_metadata_path)
    except Exception as e:
        print(f"[ERROR] Could not load files: {e}")
        return

    # 2. Ensure we are comparing the same types (Force to Int)
    sub_ids = set(sub_df['image_id'].astype(int))
    meta_ids = set(meta_df['image_id'].astype(int))

    # 3. Perform Checks
    missing_ids = sub_ids - meta_ids
    hallucinated_count = len(missing_ids)
    
    # 4. Check for duplicates (Submission should only have 1 row per image)
    duplicates = sub_df['image_id'].duplicated().sum()

    # 5. Output Results
    print(f"Total images in Submission: {len(sub_df)}")
    print(f"Total images in Master Metadata: {len(meta_df)}")
    print("-" * 35)

    if hallucinated_count == 0:
        print("✅ SUCCESS: All image IDs are valid and present in metadata.")
    else:
        print(f"❌ FAILURE: {hallucinated_count} IDs were NOT found in the master metadata!")
        print(f"Sample missing IDs: {list(missing_ids)[:10]}")

    if duplicates > 0:
        print(f"⚠️ WARNING: Your submission contains {duplicates} duplicate image_id entries!")
    else:
        print("✅ SUCCESS: No duplicate IDs found.")

if __name__ == "__main__":
    # Change these paths as needed
    SUB_FILE = r"runs\run_008_animal_clef_2026_baseline\submission.csv"
    MASTER_META = "src/images/animal-clef-2026/metadata.csv" 
    
    verify_submission(SUB_FILE, MASTER_META)