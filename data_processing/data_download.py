import os
import requests
import zipfile

# Define the URL and target paths
url = "https://cricsheet.org/downloads/all_json.zip"
# For initial coding, I used a small subset of the dataset (879 files instead of 20396)
# url = "https://cricsheet.org/downloads/tests_json.zip"
zip_path = "data/raw/cricsheet_data/all_json.zip"
extract_path = "data/raw/cricsheet_data/"

# Create the target directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Download the ZIP file
print("Downloading dataset...")
response = requests.get(url)
with open(zip_path, "wb") as f:
    f.write(response.content)
print("Download complete.")

# Unzip the contents
print("Extracting files...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction complete.")

# Delete the ZIP file
os.remove(zip_path)
print("ZIP file deleted.")