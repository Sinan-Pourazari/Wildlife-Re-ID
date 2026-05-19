import os
import glob
import time

# --- CONFIGURATION ---
ABLATIONS_DIR = "runs/ablations"
OUTPUT_FILE = "master_crash_report.txt"

# --- COLORS FOR CONSOLE ---
FOX_ORANGE   = "\033[38;5;202m"
DANGER_RED   = "\033[38;5;196m"
SUCCESS_LIME = "\033[38;5;118m"
RESET        = "\033[0m"

def main():
    print(f"{FOX_ORANGE}Crawling '{ABLATIONS_DIR}' for crash reports...{RESET}")
    
    # Recursively search for all crash_report.txt files
    search_pattern = os.path.join(ABLATIONS_DIR, "**", "crash_report.txt")
    crash_files = glob.glob(search_pattern, recursive=True)

    if not crash_files:
        print(f"\n{SUCCESS_LIME} No crash reports found! All runs executed.{RESET}")
        
        # If an old master file exists but there are no new errors, let the user know
        if os.path.exists(OUTPUT_FILE):
            print(f"*(Note: An old '{OUTPUT_FILE}' exists, but no current ablation folders contain errors.)*")
        return

    print(f"{DANGER_RED}Found {len(crash_files)} crashed runs. Compiling master log...{RESET}")

    # Write everything into the structured output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write("=================================================================\n")
        outfile.write(" ST-VGANN ABLATION CRASH REPORT\n")
        outfile.write(f" Generated on: {time.ctime()}\n")
        outfile.write(f" Total Crashed Runs: {len(crash_files)}\n")
        outfile.write("=================================================================\n\n")

        for filepath in sorted(crash_files):
            # Extract the run name from the path (e.g., runs/ablations/Exp9_H2.../crash_report.txt)
            parts = os.path.normpath(filepath).split(os.sep)
            run_name = parts[-2] if len(parts) >= 2 else "Unknown_Run"

            # Read the actual error dump
            with open(filepath, "r", encoding="utf-8") as infile:
                error_content = infile.read().strip()

            # Format the output block
            outfile.write(f" RUN: {run_name}\n")
            outfile.write(f" PATH: {filepath}\n")
            outfile.write("-" * 65 + "\n")
            outfile.write(f"{error_content}\n")
            outfile.write("=" * 65 + "\n\n")

    print(f"\n{SUCCESS_LIME} Master crash report generated successfully!{RESET}")
    print(f"--> Open '{OUTPUT_FILE}' to review the errors.")

if __name__ == "__main__":
    main()