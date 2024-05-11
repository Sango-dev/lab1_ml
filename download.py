import yaml
import os
import requests
import tarfile

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
 
def download_and_extract_archive(url, destination_folder, config=None):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
 
    # Extract the file name from the URL to use as the local filename
    filename = os.path.join(destination_folder, url.split("/")[-1])
 
    # Download the archive
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
 
    # Extract the contents of the archive
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(destination_folder)
 
    # Remove the downloaded archive file
    os.remove(filename)
 
if __name__ == "__main__":
    config_file = "config.yaml"
    config = load_config(config_file)
    archive_url = config['data']['dataset_url']
    destination_folder = config['data']['local_dir']

    download_and_extract_archive(archive_url, destination_folder, config)
    print(f"Archive extracted to {destination_folder}")
