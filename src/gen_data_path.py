import os
import glob
import _init_paths


def gen_caltech_path(root_path):
    label_path = os.path.join(root_path, 'Caltech', 'data', 'labels_with_ids')
    image_path = label_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(os.path.join(image_path, '*.png')))
    with open(os.path.join('..', 'src', 'data', 'caltech.all'), 'w') as f:
        labels = sorted(glob.glob(os.path.join(label_path, '*.txt')))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.png')
            if image in images_exist:
                print(image[22:], file=f)
    f.close()


def gen_data_path(root_path):
    mot_path = os.path.join(root_path, 'MOT17', 'images', 'train')
    seq_names = [s for s in sorted(os.listdir(mot_path)) if s.endswith('SDP')]
    with open(os.path.join('E:/FairMOT/src/data/mot17.helf'), 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(mot_path, seq_name, 'img1')
            images = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half):
                image = images[i]
                print(image[22:], file=f)
    f.close()


def gen_data_path_mot17_val(root_path):
    mot_path = os.path.join(root_path, 'MOT17', 'images', 'train')
    seq_names = [s for s in sorted(os.listdir(mot_path)) if s.endswith('SDP')]
    with open(os.path.join('E:/FairMOT/src/data/mot17.val'), 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(mot_path, seq_name, 'img1')
            images = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all):
                image = images[i]
                print(image[22:], file=f)
    f.close()


def gen_data_path_mot17_emb(root_path):
    mot_path = os.path.join(root_path, 'MOT17', 'images', 'train')
    seq_names = [s for s in sorted(os.listdir(mot_path)) if s.endswith('SDP')]
    with open(os.path.join('E:/FairMOT/src/data/mot17.emb'), 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(mot_path, seq_name, 'img1')
            images = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
            len_all = len(images)
            len_half = int(len_all / 2)
            for i in range(len_half, len_all, 3):
                image = images[i]
                print(image[22:], file=f)
    f.close()


if __name__ == '__main__':
    root = os.path.join('E:/FairMOT/dataset')
    gen_data_path_mot17_emb(root)
    gen_data_path_mot17_val(root)
    gen_data_path(root)
    gen_caltech_path(root)
