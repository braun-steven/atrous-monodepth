with open("scene_0001_train.txt", "w") as out:
    with open("kitti_train_files.txt") as file:
        for line in file:
            if "2011_09_26_drive_0001_sync" in line:
                out.write(line)

with open("scene_0001_test.txt", "w") as out:
    with open("kitti_test_files.txt") as file:
        for line in file:
            if "2011_09_26_drive_0001_sync" in line:
                out.write(line)

with open("scene_0001_val.txt", "w") as out:
    with open("kitti_val_files.txt") as file:
        for line in file:
            if "2011_09_26_drive_0001_sync" in line:
                out.write(line)
