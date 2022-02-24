import os
import numpy as np
import shutil
import random
import splitfolders

# creating train / val /test
root_dir = 'segdata/'
new_root = 'dataset/'
classes = [folderClass for folderClass in os.listdir(root_dir)]
splitfolders.ratio(root_dir, output=new_root, seed=1337, ratio=(.8, 0.1,0.1))
#
# ## creating partition of the data after shuffeling
#
# for cls in classes:
#     src = root_dir + cls  # folder to copy images from
#     print(src)
#
#     allFileNames = os.listdir(src)
#     np.random.shuffle(allFileNames)
#
#     ## here 0.75 = training ratio , (0.95-0.75) = validation ratio , (1-0.95) =
#     ##training ratio
#     train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames) * 0.75),
#                                                                                        int(len(allFileNames) * 0.95)])
#
#     # #Converting file names from array to list
#
#     train_FileNames = [src + '/' + name for name in train_FileNames]
#     val_FileNames = [src + '/' + name for name in val_FileNames]
#     test_FileNames = [src + '/' + name for name in test_FileNames]
#
#     print('Total images  : ' + cls + ' ' + str(len(allFileNames)))
#     print('Training : ' + cls + ' ' + str(len(train_FileNames)))
#     print('Validation : ' + cls + ' ' + str(len(val_FileNames)))
#     print('Testing : ' + cls + ' ' + str(len(test_FileNames)))
#
#     ## Copy pasting images to target directory
#
#     for name in train_FileNames:
#         shutil.copy(name, new_root + 'train/' + cls)
#
#     for name in val_FileNames:
#         shutil.copy(name, new_root + 'val/' + cls)
#
#     for name in test_FileNames:
#         shutil.copy(name, new_root + 'test/' + cls)