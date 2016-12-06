import glob
import sys
import csv
import argparse

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

def preprocess_images(dir_path):
    ret = []
    for study_path in glob.glob(dir_path + '/ISPY*/*/*'):
        n_images = len(glob.glob(study_path + "/*.png"))
        #print("Folder: {}, n_images: {}".format(study_path, n_images))
        ret.append(n_images)
    return ret

def parse_command_line():
    parser = argparse.ArgumentParser(description='Preprocesses data.')
    parser.add_argument('img_dir', metavar = 'img_dir', type = str,
                        help='the path to the image DOI folder.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_command_line()
    print(args)
    img_dir = args.img_dir
    img_metadata = preprocess_images(img_dir)

    plt.hist(img_metadata, bins = 'fd')
    plt.show()

    print(max(img_metadata), min(img_metadata),
            np.mean(img_metadata), np.std(img_metadata))

