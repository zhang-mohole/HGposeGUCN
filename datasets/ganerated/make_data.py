import os
import numpy as np
import random


def make_dir_path(root_path, with_object=False):
    pathes = []
    if with_object:
        with_object = root_path + "/data/withObject" # 188,050 frames in total
        for i in range(1, 184):
            end = 1025
            if i == 92:
                end = 477
            for j in range(1, end):

                pathes.append(with_object + "/{0:04d}/{1:04d}".format(i, j)) # 143,449 frames in total
    else:
        no_object = root_path + "/data/noObject"
        for i in range(1,141):
            end = 1025
            if i == 69:
                end = 217
            for j in range(1,end):
                pathes.append(no_object +"/{0:04d}/{1:04d}".format(i,j))

    return pathes

def get_data(dir_path):
    joint_3d = []
    joint_3d_global = []
    joint_2d = []
    images = []

    for path in dir_path:
        image = path+"_color_composed.png"
        images.append(image)

        value = open(path+"_joint_pos.txt").readline().strip('\n').split(',')
        value = [float(val) for val in value]
        joint_3d.append(value)

        value = open(path+"_joint2D.txt").readline().strip('\n').split(',')
        value = [float(val) for val in value]
        joint_2d.append(value)

        value = open(path+"_joint_pos_global.txt").readline().strip('\n').split(',')
        value = [float(val) for val in value]
        joint_3d_global.append(value)

    return images, joint_2d, joint_3d, joint_3d_global

if __name__ == "__main__":
    with_object = False
    dir_path = make_dir_path('/home/tlh-lxy/zmh/data/GANerated_hand/GANerated/GANeratedHands_Release', with_object=with_object)
    images, joint_2d, joint_3d, joint_3d_global = get_data(dir_path)

    total_image = len(images)
    split = total_image//10

    li = list(range(total_image))
    random.shuffle(li)

    # calculate the scale
    joint_3d_global = np.array(joint_3d_global).reshape(-1, 21, 3)
    M0 = joint_3d_global[:, 9, :]
    W = joint_3d_global[:, 0, :]
    scale = np.linalg.norm(W - M0, axis=1) # length is the scale

    # shuffle the data
    images = np.array(images)[li]
    joint_2d = np.array(joint_2d)[li]
    joint_3d = np.array(joint_3d)[li]
    scale = scale[li]

    joint_2d = joint_2d.reshape(-1,21,2)
    joint_3d = joint_3d.reshape(-1,21,3)

    # split data into train, val, and test set
    images_train = images[:split*7]
    joint_2d_trian = joint_2d[:split*7]
    joint_3d_train = joint_3d[:split*7]
    scale_train = scale[:split*7]
    np.save('images-train.npy', images_train)
    np.save('points2d-train.npy', joint_2d_trian)
    np.save('points3d-train.npy', joint_3d_train)
    np.save('scale-train.npy', scale_train)

    images_val = images[split*7:split*8]
    joint_2d_val = joint_2d[split*7:split*8]
    joint_3d_val = joint_3d[split*7:split*8]
    scale_val = scale[split*7:split*8]
    np.save('images-val.npy', images_val)
    np.save('points2d-val.npy', joint_2d_val)
    np.save('points3d-val.npy', joint_3d_val)
    np.save('scale-val.npy', scale_val)

    images_test = images[split*8:]
    joint_2d_test = joint_2d[split*8:]
    joint_3d_test = joint_3d[split*8:]
    scale_test = scale[split*8:]
    np.save('images-test.npy', images_test)
    np.save('points2d-test.npy', joint_2d_test)
    np.save('points3d-test.npy', joint_3d_test)
    np.save('scale-test.npy', scale_test)

    print('GANerated dataset done')

