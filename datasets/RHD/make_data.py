import pickle
import os
import numpy as np

def make_rhd_data(data_root, pickle_file, which_set='train'):
    joint_3d = []
    joint_2d = []
    scales = []
    images = []
    re_order = [0,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17]
    anno = pickle.load(open(os.path.join(data_root, pickle_file), 'rb'))
    anno_key = sorted(anno.keys())

    for img_name in anno_key:
        label = anno[img_name]

        image_path   = os.path.join(data_root, 'crop', img_name + '.png')
        # img = cv2.imread(image_path)
        images.append(image_path)

        original_coor2d = label['uv_original']
        original_coor2d = original_coor2d[re_order,:]
        coor3d = label['xyz_original'] # 3d coord, in meter
        coor3d = coor3d[re_order,:]
        # original_K = label['K_original']
        bbox = label['bbox']
        xy_scale = label['xy_scale']
        # kp_visible = label['visible']

        ## transform the coord and matrix
        coor2d = original_coor2d - np.tile(np.expand_dims(bbox[:2], axis=0), (21,1))
        coor2d[:, 0] *= xy_scale[0]
        coor2d[:, 1] *= xy_scale[1]
        joint_2d.append(coor2d)

        ## calculate 3d relative position
        r_coor3d = coor3d*1000 # convert to mm (from meter)
        r_coor3d = r_coor3d - r_coor3d[9,:].reshape(1,r_coor3d.shape[-1]).repeat(21, axis=0)
        M0 = r_coor3d[9, :]
        W = r_coor3d[0, :]
        # scale = np.linalg.norm(W - M0, axis=1) # length is the scale
        scale = np.linalg.norm(W - M0)
        relative_3d = r_coor3d / scale
        joint_3d.append(relative_3d)

        scales.append(scale)
    
    joint_2d = np.array(joint_2d)
    joint_3d = np.array(joint_3d)
    scales = np.array(scales)
    images = np.array(images)

    np.save('images-{}.npy'.format(which_set), images)
    np.save('points2d-{}.npy'.format(which_set), joint_2d)
    np.save('points3d-{}.npy'.format(which_set), joint_3d)
    np.save('scale-{}.npy'.format(which_set), scales)

    print(which_set, 'set processed done')
        

if __name__ == "__main__":
    train_root = '/home/zmh/datasets/dataset/RHD_published_v2/processed/training'
    train_pickle = 'RHD_training.pickle'
    make_rhd_data(train_root, train_pickle, which_set='train')
    test_root = '/home/zmh/datasets/dataset/RHD_published_v2/processed/evaluation'
    test_pickle = 'RHD_evaluation.pickle'
    make_rhd_data(test_root, test_pickle, which_set='test')
    