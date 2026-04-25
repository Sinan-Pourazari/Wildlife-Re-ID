import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n[ Pipeline Initializer ] Loading Master CSV: {args.csv_path}")

    # 1. Load metadata
    df = pd.read_csv(args.csv_path, low_memory=False)

    # ========================================================
    # 2. STRICT COMPETITION TEST EXTRACTION
    # ========================================================
    comp_test_df = pd.DataFrame()
    if 'split' in df.columns and 'source_domain' in df.columns:
        # Use bitwise & with parentheses for Pandas!
        comp_mask = (df['split'] == 'test') & (df['source_domain'] == 'animalclef')
        comp_test_df = df[comp_mask].copy()
        
        # Safely remove them from the main dataframe without deleting reid10k
        df = df[~comp_mask].reset_index(drop=True)
        print(f"--> Extracted {len(comp_test_df)} competition test points. Remaining train pool: {len(df)}")

    if 'identity' not in df.columns:
        df['identity'] = df['animal_id'].astype(str)
    else:
        df['identity'] = df['identity'].astype(str)

    # 3. Filter by target dataset (e.g., ATRW, SMALST)
    if args.datasets:
        target_datasets = [d.lower() for d in args.datasets]
        if 'dataset' in df.columns:
            df = df[df['dataset'].str.lower().isin(target_datasets)].reset_index(drop=True)
            print(f"--> Filtered to target datasets: {target_datasets}. Remaining: {len(df)}")
        else:
            print(f"--> WARNING: 'dataset' column not found in CSV. Ignoring filter.")
            
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
    # 4. STRICT DISJOINT SPLIT (Restricted to Local Test Domain)
    # ========================================================
    print("--> Executing Strict Disjoint Split (16% identities held out)...")
    
    # Restrict the test pool to ONLY the requested domain (e.g., animalclef)
    if args.local_test_domain and 'source_domain' in df.columns:
        print(f"--> Restricting local test split strictly to domain: {args.local_test_domain}")
        valid_test_rows = df[df['source_domain'] == args.local_test_domain]
        test_pool_ids = valid_test_rows['global_label'].unique()
    else:
        test_pool_ids = df['global_label'].unique()

    # Calculate 16% of the restricted pool
    import numpy as np
    num_test_ids = int(len(test_pool_ids) * 0.16)
    
    # Sample the test identities
    np.random.seed(42)
    test_ids = np.random.choice(test_pool_ids, num_test_ids, replace=False)
    
    # Split the dataframe
    test_df = df[df['global_label'].isin(test_ids)].copy()
    train_df = df[~df['global_label'].isin(test_ids)].copy()

    # 5. Generate contiguous PyG labels (Train set only)
    train_le = LabelEncoder()
    species_le = LabelEncoder()
    
    train_df['contiguous_label'] = train_le.fit_transform(train_df['global_label'])
    
    if 'species' in df.columns:
        train_df['species_label'] = species_le.fit_transform(train_df['species'])
        # Handle unseen species safely in the test set
        test_species_mask = test_df['species'].isin(species_le.classes_)
        test_df['species_label'] = -1
        test_df.loc[test_species_mask, 'species_label'] = species_le.transform(test_df.loc[test_species_mask, 'species'])
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
    print(f"Train size: {len(train_df)} images (Includes ALL allowed datasets)")
    print(f"Test size:  {len(test_df)} images ({len(test_ids)} identities STRICTLY from {args.local_test_domain})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--local_test_domain", type=str, default="animalclef", help="Force local test split to ONLY use data from this domain.")
    main(parser.parse_args())