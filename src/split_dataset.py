import splitfolders

# creating train / val /test
root_dir = 'segdata'
new_root = 'dataset'

splitfolders.ratio(root_dir, output=new_root, seed=1337, ratio=(.8, 0.1,0.1))