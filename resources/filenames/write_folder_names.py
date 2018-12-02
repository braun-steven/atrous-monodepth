filenames = ['cityscapes_test_files.txt', 'cityscapes_train_files.txt','cityscapes_val_files.txt']

for file in filenames:
    with (open(file,'r+')) as f:
        with open(file[:-4] + '_folder.txt', 'w+') as d:
            content = f.readlines()
            for line in content:
                files = line.split()
                d.write('left/' + files[0] + ' ' + 'right/' + files[1] + '\n')

