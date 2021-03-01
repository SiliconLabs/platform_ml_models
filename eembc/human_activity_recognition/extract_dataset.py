import urllib
import zipfile

#url = "http://www.gutenberg.lib.md.us/4/8/8/2/48824/48824-8.zip"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
extract_dir = "data"

zip_path, _ = urllib.request.urlretrieve(url)

with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(extract_dir)


