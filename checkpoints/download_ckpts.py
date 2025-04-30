import os
import requests
from tqdm import tqdm


def download_file(url, dest_folder="."):
    os.makedirs(dest_folder, exist_ok=True)
    local_filename = os.path.join(dest_folder, url.split("/")[-1])

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(local_filename, "wb") as f, tqdm(
            desc=local_filename,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"✅ Downloaded: {local_filename}")
    return local_filename


# Base URL for SAM 2.1 checkpoints
base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# List of checkpoints
checkpoints = [
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt",
]

# Download all checkpoints
for ckpt in checkpoints:
    try:
        url = f"{base_url}/{ckpt}"
        download_file(url, dest_folder="checkpoints")
    except Exception as e:
        print(f"❌ Failed to download {ckpt}: {e}")
