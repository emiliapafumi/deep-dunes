import os
import gdown

DATASET_IDS = {
    'dune-wv': '15B8VujTcUF7urSTjqXACylxuId5KVQhA',
    'dune-uav': '11hjNhpcm_5OuZlQ5pqVJgPcgEFRz_QJx',
    'dune-ge': '1vGDn4wGSZ4YqEHbHmHGr2r8lTOq0rkph',
    'dune-air': '1jaj01lNvus4WupYBKa4Z3nAanwGfij1d'
}

def setup(directory_name: str):
    if directory_name not in DATASET_IDS:
        print(f"❌ Folder not found: {directory_name}")
        raise Exception(f"Invalid folder: {directory_name}")
        return

    dir = f"deep-dunes-data/{directory_name}"
    if os.path.isdir(dir):
        print(f"✅ Directory exists at {dir}. Skipping download")
        return
    
    create(name=dir)

    print(f"⬇️ Downloading '{directory_name}'...")
    try:
        gdown.download_folder(
            f'https://drive.google.com/drive/folders/{DATASET_IDS[directory_name]}',
            output=dir,
            quiet=True,
            use_cookies=False
        )
        print(f"✅ Successfully downloaded: {directory_name}")
        return
    except Exception as e:
        print(f"❌ Direct download failed: {e}")
        raise e

def create(name: str):
    try:
        os.makedirs(name, exist_ok=True)
    except PermissionError:
        print(f"Permission denied: Unable to create '{name}'.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
