with open("scene_0001_train.txt", 'w') as out:
    with open("kitti_train_files.txt") as file:
        for line in file:
            if "drive_0001" in line:
                out.write(line)

with open("scene_0001_test.txt", 'w') as out:
    with open("kitti_test_files.txt") as file:
        for line in file:
            if "drive_0001" in line:
                out.write(line)
                
with open("scene_0001_val.txt", 'w') as out:
    with open("kitti_val_files.txt") as file:
        for line in file:
            if "drive_0001" in line:
                out.write(line)

