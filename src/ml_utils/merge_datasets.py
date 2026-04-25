import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reid_csv", required=True)
    parser.add_argument("--clef_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df_reid = pd.read_csv(args.reid_csv, low_memory=False)
    df_clef = pd.read_csv(args.clef_csv, low_memory=False)

    # ==========================================================
    # EDGE CASE HANDLING: Deduplicate SeaTurtleID2022
    # We drop it from reid-10k to strictly favor the AnimalCLEF version
    # ==========================================================
    if 'dataset' in df_reid.columns:
        original_reid_len = len(df_reid)
        df_reid = df_reid[df_reid['dataset'].str.lower() != 'seaturtleid2022'].copy()
        dropped_count = original_reid_len - len(df_reid)
        if dropped_count > 0:
            print(f"--> [DEDUPLICATION] Dropped {dropped_count} 'SeaTurtleID2022' images from ReID-10k.")

    # 1. Update paths
    df_reid['path'] = 'reid-10k/' + df_reid['path'].astype(str)
    df_clef['path'] = 'animal-clef-2026/' + df_clef['path'].astype(str)

    # 2. INJECT BULLETPROOF DOMAIN TAGS
    df_reid['source_domain'] = 'reid10k'
    df_clef['source_domain'] = 'animalclef'

    # 3. Merge and save
    merged_df = pd.concat([df_reid, df_clef], ignore_index=True)
    merged_df.to_csv(args.out_csv, index=False)
    print(f"--> Merged datasets! Total: {len(merged_df)} images.")

if __name__ == "__main__":
    main()