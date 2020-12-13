import pickle
import os
import numpy as np

def make_data(STB_root, istrain): 
    # name_set = ['B1Counting', 'B2Counting','B3Counting', 'B4Counting','B5Counting', 'B6Counting']
    train_set = ['B2Counting','B3Counting', 'B4Counting','B5Counting', 'B6Counting',
                 'B2Random','B3Random', 'B4Random','B5Random', 'B6Random']
    test_set = ['B1Counting', 'B1Random']
    if istrain:
        sets = train_set
        suffix = 'train'
    else:
        sets = test_set
        suffix = 'test'
    db = pickle.load(open(os.path.join(STB_root, 'STB_BB.pickle'), 'rb'))
    path = []
    coor3ds = []
    coor2ds = []
    scale = []

    re_order = [0,17,18,19,20,13,14,15,16,9,10,11,12,5,6,7,8,1,2,3,4]

    for name in sets:
        coor3d = db[name]['bb']['coor3d'] # [1500, 21, 3]
        r_coor3d = coor3d - coor3d[:,0,:].reshape(coor3d.shape[0],1,coor3d.shape[2]).repeat(21, axis=1)
        uv = db[name]['bb']['coor2d']

        M0 = coor3d[:, 9, :] # [1500, 3]
        W = coor3d[:, 0, :]
        real_len = np.linalg.norm(W - M0, axis=1) # [1500]

        for ix in range(1500):
            relative_3d = r_coor3d[ix] / real_len[ix] 
            coor2d = uv[ix]
            image_path = os.path.join(STB_root, 'cropped_images', 
                                      name+'_BB', 'BB_left_' + str(ix) + '.png')
            
            path.append(image_path)
            coor3ds.append(relative_3d)
            coor2ds.append(coor2d)
            scale.append(real_len[ix])

    print(len(coor2ds), len(coor3ds))
    path = np.array(path)
    joint_2d = np.array(coor2ds)
    joint_2d = joint_2d[:, re_order, :] # reorder the joints
    joint_3d = np.array(coor3ds)
    joint_3d = joint_3d[:, re_order, :]
    print(joint_2d.shape, joint_3d.shape)
    scale = np.array(scale)

    np.save('images-{}.npy'.format(suffix), path)
    np.save('points2d-{}.npy'.format(suffix), joint_2d)
    np.save('points3d-{}.npy'.format(suffix), joint_3d)
    np.save('scale-{}.npy'.format(suffix), scale)


if __name__ == "__main__":
    stb_root = '/home/zmh/datasets/hand-pose-STB/' # '/home/tlh-lxy/zmh/data/STB'
    make_data(stb_root, istrain=True)
    make_data(stb_root, istrain=False)

