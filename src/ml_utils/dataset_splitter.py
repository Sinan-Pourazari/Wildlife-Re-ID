import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n[ Pipeline Initializer ] Loading Master CSV: {args.csv_path}")

    # 1. Load your massive metadata
    df = pd.read_csv(args.csv_path, low_memory=False)

    # 2. FILTER OUT COMPETITION TEST DATA
    # We only want images that actually have labels
    comp_test_df = pd.DataFrame()
    if 'split' in df.columns:
        comp_test_df = df[df['split'] == 'test'].copy()
        df = df[df['split'] == 'train'].reset_index(drop=True)
        print(f"--> Filtered strictly to 'train' split. Remaining: {len(df)}")

    if 'identity' not in df.columns:
        df['identity'] = df['animal_id'].astype(str)
    else:
        df['identity'] = df['identity'].astype(str)

    # 3. Filter by target species (e.g., lynx, lizard)
    if args.datasets:
        target_species = [s.lower() for s in args.datasets]
        if 'species' in df.columns:
            df = df[df['species'].str.lower().isin(target_species)].reset_index(drop=True)
            print(f"--> Filtered to target species: {target_species}. Remaining: {len(df)}")

    # Drop unknowns and singletons
    df = df[df['identity'] != 'unknown'].reset_index(drop=True)
    df = df.dropna(subset=['identity']).reset_index(drop=True)
    
    if 'dataset' in df.columns:
        df['global_identity'] = df['dataset'] + "_" + df['identity']
    else:
        df['global_identity'] = df['identity']
        
    counts = df['global_identity'].value_counts()
    keep_ids = counts[counts > 1].index
    df = df[df['global_identity'].isin(keep_ids)].reset_index(drop=True)

    df['global_label'] = LabelEncoder().fit_transform(df['global_identity'])

    # ========================================================
    # 4. STRICT DISJOINT SPLIT (Pure Open-Set / Zero-Shot)
    # ========================================================
    print("--> Executing Strict Disjoint Split (16% identities held out)...")
    unique_identities = df['global_label'].unique()
    
    # Isolate 16% of IDENTITIES completely
    train_ids, test_ids = train_test_split(unique_identities, test_size=0.16, random_state=42)
    
    train_df = df[df['global_label'].isin(train_ids)].copy()
    test_df = df[df['global_label'].isin(test_ids)].copy()

    # 5. Generate contiguous PyG labels (Train set only)
    train_le = LabelEncoder()
    species_le = LabelEncoder()
    
    train_df['contiguous_label'] = train_le.fit_transform(train_df['global_label'])
    
    if 'species' in df.columns:
        train_df['species_label'] = species_le.fit_transform(train_df['species'])
        test_df['species_label'] = species_le.transform(test_df['species'])
    else:
        train_df['species_label'] = 0
        test_df['species_label'] = 0
        
    test_df['contiguous_label'] = -1 

    # 6. Save the final datasets
    train_path = os.path.join(args.save_dir, "train_split.csv")
    test_path = os.path.join(args.save_dir, "test_split.csv")
    pipeline_path = os.path.join(args.save_dir, "pipeline_metadata.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    pd.concat([train_df, test_df], ignore_index=True).to_csv(pipeline_path, index=False)

    # 7. Format and save the actual competition test set
    if not comp_test_df.empty:
        comp_test_df['global_label'] = -1
        comp_test_df['contiguous_label'] = -1
        comp_test_df['species_label'] = -1
        comp_test_path = os.path.join(args.save_dir, "competition_test.csv")
        comp_test_df.to_csv(comp_test_path, index=False)
        print(f"Competition Test size: {len(comp_test_df)} images ready for submission.")

    print(f"\n[ Split Complete ]")
    print(f"Train size: {len(train_df)} images ({len(train_ids)} identities)")
    print(f"Test size:  {len(test_df)} images ({len(test_ids)} STRICTLY UNSEEN identities)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    main(parser.parse_args())