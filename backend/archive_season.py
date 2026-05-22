import shutil
from pathlib import Path
from datetime import datetime

def archive_season(year=None):
    if year is None:
        year = datetime.now().year
    
    history_dir = Path("data/history")
    archive_dir = Path("data/archives")
    
    if not history_dir.exists():
        print(f"No history directory found")
        return
    
    archive_dir.mkdir(exist_ok=True)
    
    history_files = list(history_dir.glob("*.json"))
    if not history_files:
        print(f"No history files to archive")
        return
    
    zip_path = archive_dir / f"season_{year}"
    print(f"Archiving {len(history_files)} files to {zip_path}.zip")
    
    shutil.make_archive(str(zip_path), 'zip', history_dir)
    
    for file in history_files:
        file.unlink()
    
    print(f"Archive complete. Cleared history directory.")

if __name__ == "__main__":
    archive_season()
