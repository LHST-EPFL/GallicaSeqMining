# %%
import re
import os
import pandas as pd
import smbclient
import gzip
import shutil
# deactivating warnings
import warnings

warnings.filterwarnings("ignore")

# %%
# smb session

LOCAL_PATH = r"Data"

# smb connection
smbclient.register_session(server=SERVER, username=USERNAME, password=PASSWORD)

# get the directories
directories = smbclient.listdir(path=PATH)
# remove ds_store and readme
directories = [x for x in directories if x not in [".DS_Store", "readme.txt"]]
# sort the directories by the number at the start of the name
pattern = re.compile(r"(\d+)")
directories = sorted(directories, key=lambda x: int(
    pattern.search(x).group(1)))
print(directories)

# %%
# get all the file names in the directories and store them in a pandas dataframe following this format: directory, file_name, file_path
df_files = pd.DataFrame(columns=["directory", "file_name", "file_path"])
for directory in directories:
    files = smbclient.listdir(path=f"{PATH}/{directory}")
    files = [x for x in files if x not in [".DS_Store", "readme.txt"]]
    files = pd.DataFrame(files, columns=["file_name"])
    files["directory"] = directory
    files["file_path"] = files["file_name"].apply(
        lambda x: f"{PATH}/{directory}/{x}")
    files["local_path"] = files["file_name"].apply(
        lambda x: f"{LOCAL_PATH}/{directory}/{x}")
    df_files = df_files.append(files, ignore_index=True)

# filter the files that are not log files
df_files = df_files[df_files["file_name"].str.contains("log")]

# extract the file number from the file name
df_files["file_number"] = df_files["file_name"].apply(
    lambda x: int(re.search(r"(\d+)", x).group(1)))
# extract the directory number from the directory name
pattern = re.compile(r"(\d+)")
df_files["directory_number"] = df_files["directory"].apply(
    lambda x:  int(pattern.search(x).group(1)))

# sort by directory number than by file number
df_files = df_files.sort_values(by=["directory_number", "file_number"])

# %%
# copy files from the smb server to the local machine
for index, row in df_files.iterrows():
    if not os.path.exists(row["local_path"]):
        # write file to local machine
        with smbclient.open_file(row["file_path"], mode='rb') as remote_file:
            with open(row["local_path"], 'wb') as local_file:
                local_file.write(remote_file.read())
        print(f"{row['file_name']} successfully copied to {row['local_path']}")
    else:
        print(f"{row['file_name']} already exists in {row['local_path']}")
print("the data was copied successfully :)")

# %%
# Get the local_path of the files whose local path does not end with .gz
files_to_compress = df_files[df_files['local_path'].str.endswith(
    '.gz') == False]['local_path'].tolist()

# Check if there are any files to compress
if files_to_compress:
    print(files_to_compress)

    # Compress the files whose local path does not end with .gz
    for file_path in files_to_compress:
        # Compress the file directly
        compressed_file_path = file_path + '.gz'
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Delete the original file
        os.remove(file_path)

    # Update df_files to reflect the compressed files
    df_files.loc[df_files['local_path'].isin(files_to_compress), 'local_path'] = df_files.loc[df_files['local_path'].isin(
        files_to_compress), 'local_path'] + '.gz'
else:
    print("No files found that need to be compressed.")

# %%
# if csv directory does not exist, create it
if not os.path.exists("temp_data"):
    os.makedirs("temp_data")
# store csv file
df_files.to_csv("temp_data/files.csv", index=False)


