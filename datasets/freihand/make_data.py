import os
import numpy as np
import random
import fh_utils

def make_freihand_data(no_obj_li, with_obj_li, need_obj=False):
    data_root = '/home/tlh-lxy/zmh/data/FreiHAND'
    uv_li = []
    set_name = 'training'
    K_list, _, xyz_list = fh_utils.load_db_annotation(data_root, set_name=set_name)
    for i, xyz in enumerate(xyz_list):
        uv = fh_utils.projectPoints(xyz, K_list[i]) # (21,2) np array
        uv_li.append(uv)

    uv = np.array(uv_li) # (32560, 21, 2)
    uv = np.tile(uv, (4,1,1)) # (32560*4, 21, 2)
    xyz_li = np.array(xyz_list) * 1000 # convert to mm (from meter)
    M0 = xyz_li[:, 9, :]
    W = xyz_li[:, 0, :]
    scale = np.linalg.norm(W - M0, axis=1) # length is the scale
    xyz_li = np.tile(xyz_li, (4,1,1))
    scale = np.tile(scale, 4)
    img_li = [os.path.join(data_root, set_name, 'rgb','%08d.jpg' % i) for i in range(32560*4)]
    img_li = np.array(img_li)

    r_coor3d = xyz_li - xyz_li[:,9,:].reshape(xyz_li.shape[0],1,xyz_li.shape[2]).repeat(21, axis=1)
    relative_3d = r_coor3d / np.tile(scale.reshape(-1,1,1), (1,21,3))

    no_obj_uv = uv[no_obj_li]
    no_obj_img = img_li[no_obj_li]
    no_obj_3d = relative_3d[no_obj_li]
    no_obj_scale = scale[no_obj_li]

    with_obj_uv = uv[with_obj_li]
    with_obj_img = img_li[with_obj_li]
    with_obj_3d = relative_3d[with_obj_li]
    with_obj_scale = scale[with_obj_li]

    if need_obj:
        images = with_obj_img
        joint_2d = with_obj_uv
        joint_3d = with_obj_3d
        scale = with_obj_scale

        total_image = images.shape[0]
    else:
        images = no_obj_img
        joint_2d = no_obj_uv
        joint_3d = no_obj_3d
        scale = no_obj_scale

        total_image = images.shape[0]
    
    split = total_image//10

    li = list(range(total_image))
    random.shuffle(li)

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

    print('freihand dataset done')

def separate_set(no_obj_root, with_obj_root):
    no_obj_li = []
    with_obj_li = []
    no_obj_files = os.listdir(no_obj_root)
    for file_name in no_obj_files:
        if file_name.endswith('.jpg'):
            img_id = int(file_name[:-4])
            no_obj_li.append(img_id)

    with_obj_files = os.listdir(with_obj_root)
    for file_name in with_obj_files:
        if file_name.endswith('.jpg'):
            img_id = int(file_name[:-4])
            with_obj_li.append(img_id)

    no_obj_li.sort()
    with_obj_li.sort()

    l_no_obj = []
    l_with_obj = []

    for i in range(0,4):
        for ix in no_obj_li:
            l_no_obj.append(ix+i*32560)

    for i in range(0,4):
        for ix in with_obj_li:
            l_with_obj.append(ix+i*32560)

    print(l_with_obj)

    return l_no_obj, l_with_obj


if __name__ == "__main__":
    no_obj_li, with_obj_li = separate_set('/home/tlh-lxy/zmh/data/FreiHAND/no-object', '/home/tlh-lxy/zmh/data/FreiHAND/with-object')
    make_freihand_data(no_obj_li, with_obj_li, need_obj=False)