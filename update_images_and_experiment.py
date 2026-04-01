import pandas as pd
import glob
import shutil
from pathlib import Path


# --- Your original helper function ---
def find_photo(folder, pic_num):
    folder = Path(folder)
    pic_num = int(pic_num)

    matches = []
    for ext in ["jpg", "jpeg", "png"]:
        matches.extend(glob.glob(str(folder / f"*{pic_num:04d}.{ext}")))
        matches.extend(glob.glob(str(folder / f"*{pic_num:04d}.{ext.upper()}")))

    unique_matches = list(dict.fromkeys(matches))

    if not unique_matches:
        raise FileNotFoundError(f"No photo for {pic_num}")

    if len(unique_matches) > 1:
        raise ValueError(f"Multiple matches for {pic_num}: {unique_matches}")

    return unique_matches[0]


# --- The Sync Function ---
def sync_project_state(csv_path, top_folder, side_folder, removed_list_path, removed_top_folder, removed_side_folder,
                       output_csv_path):
    # 1. Load the list of bad side image filenames you shared
    with open(removed_list_path, "r") as f:
        removed_filenames = set(line.strip() for line in f if line.strip())

    # 2. Read the CSV
    data = pd.read_csv(csv_path)
    data["pic_top"] = pd.to_numeric(data["pic_top"], errors="coerce")
    data["pic_side"] = pd.to_numeric(data["pic_side"], errors="coerce")

    # 3. Figure out which rows to remove
    def is_marked_for_removal(pic_side_num):
        if pd.isna(pic_side_num):
            return False
        try:
            # Find the photo in the collaborator's active side folder
            photo_path = Path(find_photo(side_folder, pic_side_num))
            # Check if its filename is in the text file
            return photo_path.name in removed_filenames
        except FileNotFoundError:
            return False  # If they already removed it manually, ignore

    print("Checking which images need to be removed...")
    mask_to_remove = data["pic_side"].apply(is_marked_for_removal)
    rows_to_remove = data[mask_to_remove]

    if rows_to_remove.empty:
        print("No images found that need to be removed. Everything is up to date!")
        return

    print(f"Found {len(rows_to_remove)} rows to remove. Syncing folders...")

    # 4. Create the target folders for the removed images
    Path(removed_top_folder).mkdir(parents=True, exist_ok=True)
    Path(removed_side_folder).mkdir(parents=True, exist_ok=True)

    # Move the corresponding top and side images so their folders match yours
    for idx, row in rows_to_remove.iterrows():
        # Move Side Image
        if pd.notna(row["pic_side"]):
            try:
                side_img_path = Path(find_photo(side_folder, row["pic_side"]))
                shutil.move(str(side_img_path), str(Path(removed_side_folder) / side_img_path.name))
                print(f"  Moved side: {side_img_path.name}")
            except FileNotFoundError:
                pass  # Already moved

        # Move Top Image
        if pd.notna(row["pic_top"]):
            try:
                top_img_path = Path(find_photo(top_folder, row["pic_top"]))
                shutil.move(str(top_img_path), str(Path(removed_top_folder) / top_img_path.name))
                print(f"  Moved top:  {top_img_path.name}")
            except FileNotFoundError:
                pass

    # 5. Save the updated CSV
    data_cleaned = data[~mask_to_remove].copy()
    data_cleaned.to_csv(output_csv_path, index=False)
    print(f"\nDone! Updated CSV saved to: {output_csv_path} (Remaining rows: {len(data_cleaned)})")


if __name__ == "__main__":
    # Collaborators should ensure these paths point to their local project folders
    sync_project_state(
        csv_path="experiments/experiments_1_110.csv",
        top_folder="photos/top_view_images",
        side_folder="photos/side_view_images",
        removed_list_path="removed_side_photos.txt",
        removed_top_folder="photos/top_view_images/exclude",
        removed_side_folder="photos/side_view_images/exclude",
        output_csv_path="experiments_1_110_cleaned.csv"
    )