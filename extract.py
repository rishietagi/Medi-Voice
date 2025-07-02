import tarfile
import os

input_path = r"C:\Users\Rishi S Etagi\Downloads\archive (2)\20200803\20200803.tar.gz.aa"
output_dir = "extracted_audio"

with tarfile.open(input_path, "r:gz") as tar:
    tar.extractall(path=output_dir)

print("Extraction complete.")
