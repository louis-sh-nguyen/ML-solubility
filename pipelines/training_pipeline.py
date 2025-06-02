# pipelines/training_pipeline.py

import os
import time
import subprocess


def has_new_data(new_dir: str, processed_flag: str) -> bool:
    """
    Determines if any CSV in new_dir is newer than the processing flag file.
    If the flag does not exist, return True.
    """
    if not os.path.isdir(new_dir):
        return False

    if not os.path.exists(processed_flag):
        return True

    flag_time = os.path.getmtime(processed_flag)
    for fname in os.listdir(new_dir):
        if fname.endswith(".csv"):
            full_path = os.path.join(new_dir, fname)
            if os.path.getmtime(full_path) > flag_time:
                return True
    return False


def update_processed_flag(flag_path: str):
    """
    Creates or updates the timestamp on the flag file.
    """
    with open(flag_path, 'a'):
        os.utime(flag_path, None)


def run_pipeline():
    """
    1. Check for new data in data/new/
    2. If new data exists, run train.py
    3. Always run evaluate.py
    4. Update processed flag if training occurred
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    new_data_dir = os.path.join(project_root, "data/new")
    processed_flag = os.path.join(project_root, "data/processed/.last_processed")

    if has_new_data(new_data_dir, processed_flag):
        print("New data detected—starting training.")
        subprocess.run(["python", os.path.join(project_root, "src/train.py")], check=True)
        update_processed_flag(processed_flag)
    else:
        print("No new data detected. Skipping training.")

    print("Running evaluation.")
    subprocess.run(["python", os.path.join(project_root, "src/evaluate.py")], check=True)


if __name__ == "__main__":
    # Poll every hour (3600 seconds)
    POLL_INTERVAL = 3600
    while True:
        run_pipeline()
        time.sleep(POLL_INTERVAL)
