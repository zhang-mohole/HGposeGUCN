import pickle
import os
import numpy as np

def make_data(STB_root):
    name_set = ['B1Counting', 'B2Counting','B3Counting', 'B4Counting','B5Counting', 'B6Counting']
    db = pickle.load(open(os.path.join(STB_root, 'STB_SK.pickle'), 'rb'))
    path = []
    coor3ds = []
    coor2ds = []
    scale = []

    re_order = [0,17,18,19,20,13,14,15,16,9,10,11,12,5,6,7,8,1,2,3,4]

    for name in name_set:
        coor3d = db[name]['sk']['coor3d'] # [1500, 21, 3]
        r_coor3d = coor3d - coor3d[:,0,:].reshape(coor3d.shape[0],1,coor3d.shape[2]).repeat(21, axis=1)
        matrix = db[name]['sk']['matrix']

        M0 = coor3d[:, 9, :] # [1500, 3]
        W = coor3d[:, 0, :]
        real_len = np.linalg.norm(W - M0, axis=1) # [1500]
        # relative_3d = coor3d[ix] / np.tile(real_len.reshape(-1,1), (21*3)).reshape(-1, 21, 3)

        for ix in range(1500):
            relative_3d = r_coor3d[ix] / real_len[ix] 
            coor2d = np.matmul(matrix[ix], coor3d[ix].T).T
            coor2d_d = coor2d[:, -1:]
            coor2d[:,:2] /= coor2d_d
            image_path = os.path.join(STB_root, 'cropped_images', 
                                      name+'_SK', 'SK_color_' + str(ix) + '.png')
            
            path.append(image_path)
            coor3ds.append(relative_3d)
            coor2ds.append(coor2d[:,:2])
            scale.append(real_len[ix])

    print(len(coor2ds), len(coor3ds))
    path = np.array(path)
    joint_2d = np.array(coor2ds)
    joint_2d = joint_2d[:, re_order, :] # reorder the joints
    joint_3d = np.array(coor3ds)
    joint_3d = joint_3d[:, re_order, :]
    print(joint_2d.shape, joint_3d.shape)
    scale = np.array(scale)

    np.save('images-train.npy', path[1500*2:])
    np.save('points2d-train.npy', joint_2d[1500*2:])
    np.save('points3d-train.npy', joint_3d[1500*2:])
    np.save('scale-train.npy', scale[1500*2:])

    np.save('images-val.npy', path[1500:1500*2])
    np.save('points2d-val.npy', joint_2d[1500:1500*2])
    np.save('points3d-val.npy', joint_3d[1500:1500*2])
    np.save('scale-val.npy', scale[1500:1500*2])

    np.save('images-test.npy', path[:1500])
    np.save('points2d-test.npy', joint_2d[:1500])
    np.save('points3d-test.npy', joint_3d[:1500])

    np.save('scale-test.npy', scale[:1500])


if __name__ == "__main__":
    stb_root = '/home/zmh/datasets/hand-pose-STB/' # '/home/tlh-lxy/zmh/data/STB'
    make_data(stb_root)

