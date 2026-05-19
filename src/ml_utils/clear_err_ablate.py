import os
import glob

# --- CONFIGURATION ---
ABLATIONS_DIR = "runs/ablations"
TARGET_FILE = "crash_report.txt"
MASTER_REPORT = "master_crash_report.txt"

# --- COLORS FOR CONSOLE ---
FOX_ORANGE   = "\033[38;5;202m"
SUCCESS_LIME = "\033[38;5;118m"
DIRT_BROWN   = "\033[38;5;94m"
RESET        = "\033[0m"

def main():
    print(f"{FOX_ORANGE}Crawling '{ABLATIONS_DIR}' to delete {TARGET_FILE} files...{RESET}")
    
    # Recursively find all crash_report.txt files in the subfolders
    search_pattern = os.path.join(ABLATIONS_DIR, "**", TARGET_FILE)
    crash_files = glob.glob(search_pattern, recursive=True)

    deleted_count = 0

    if not crash_files:
        print(f"--> No '{TARGET_FILE}' files found in subdirectories.")
    else:
        print(f"--> Found {len(crash_files)} local crash reports. Deleting...")
        for filepath in crash_files:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"{DIRT_BROWN}[!] Failed to delete {filepath}: {e}{RESET}")

    # Also delete the master report if it exists in the root directory
    if os.path.exists(MASTER_REPORT):
        try:
            os.remove(MASTER_REPORT)
            print(f"--> Deleted '{MASTER_REPORT}'.")
            deleted_count += 1
        except Exception as e:
            print(f"{DIRT_BROWN}[!] Failed to delete {MASTER_REPORT}: {e}{RESET}")

    if deleted_count > 0:
        print(f"\n{SUCCESS_LIME} Successfully deleted {deleted_count} error logs!{RESET}")
    else:
        print(f"\n{SUCCESS_LIME} Workspace is already clean!{RESET}")
        

if __name__ == "__main__":
    main()