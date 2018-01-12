import numpy as np
import os
import sys
from skimage import io
from os import listdir
from os.path import isfile, join

data_path = sys.argv[1]
recon_path = sys.argv[2]
data_path = data_path+'/'

onlyfiles = [join(data_path,f).replace('//','/') for f in listdir(data_path) if isfile(join(data_path, f))]
pics = []

for filename in onlyfiles:
    mercury = io.imread(filename).reshape(-1)
    pics.append(mercury)

pics = np.array(pics)
mean_pic = pics.mean(axis=0)
mean_pic = mean_pic.reshape(1,-1)
pics = pics-mean_pic

U, s, V = np.linalg.svd(pics,full_matrices=False)
forth = V[:4]
recon = io.imread(recon_path)
recon = np.array(recon)
recon = recon.reshape(1,-1)
weight = np.dot(recon-mean_pic,forth.T)
re_pic = np.dot(weight,forth)+mean_pic

def scale(re_pic):
    re_pic -= np.min(re_pic)
    re_pic /= np.max(re_pic)
    re_pic = re_pic*255
    return re_pic

re_pic = scale(re_pic)

io.imsave('reconstruction.jpg',re_pic.reshape(600,600,3).astype(np.uint8))
