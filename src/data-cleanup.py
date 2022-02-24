import os

folder = 'segdata'
for classfolder in os.listdir(folder):
    length = len(os.listdir(folder + '/' + classfolder))
    if length <= 0:
        os.rmdir(folder + '/' + classfolder)