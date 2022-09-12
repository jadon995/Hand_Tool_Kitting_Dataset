import os
import glob
import shutil

path = '/home/jadon/Code/SEaT/dataset/ABC_CHUNK/*.obj'
files = glob.glob(path)

for file in files:
    filename = os.path.basename(file)
    mainfolder = os.path.dirname(file)
    subfolder = filename.split('_')[0]
    
    newfile = os.path.join(mainfolder, subfolder)
    newfile = os.path.join(newfile, filename)
    # print(filename, subfolder, mainfolder)

    print(file)
    print(newfile)
    shutil.move(file, newfile)

# print(files)