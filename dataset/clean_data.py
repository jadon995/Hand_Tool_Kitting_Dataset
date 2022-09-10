import os
import shutil

name_txt = "train.txt"
train_data = set()
with open(name_txt) as f:
    for line in f:
        path = os.path.dirname(line)
        data = os.path.basename(path)
        train_data.add(data)

folder_name = "ABC_CHUNK"
folders = os.listdir(folder_name)

for folder in folders:
    if folder not in train_data:
        folder_path = os.path.join(folder_name, folder)
        print(folder_path)
        if (os.path.exists(folder_path)):
            # os.rmdir(folder_path)
            shutil.rmtree(folder_path)
