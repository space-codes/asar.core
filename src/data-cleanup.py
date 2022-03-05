import os
import shutil

folder = 'segdata'
# Delete these folders
for classfolder in os.listdir(folder):
    length = len(os.listdir(folder + '/' + classfolder))
    deleted_folders = ["k", "mislabel", "ءء", "ءل", "ءه", "ءية"]
    if length <= 0 or (classfolder in deleted_folders):
        print("deleting"+ classfolder +"...")
        shutil.rmtree(folder + '/' + classfolder)


# special case for ا
images = os.listdir(folder + '/' + 'ا')
images = images[:20]
for image in images:
    print("copying " + image)
    shutil.copy(folder + '/' + 'ا' + '/' + image, folder + '/' + "اا")


## After transferred delete previous one and rename new one
shutil.rmtree(folder + '/' + 'ا')
os.rename(folder + '/' +'اا', folder + '/' + 'ا')