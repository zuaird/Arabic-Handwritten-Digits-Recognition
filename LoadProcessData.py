import numpy as np
import os
from PIL import Image
import requests
import patoolib

X = [np.zeros(7200)]
Y = np.zeros(60011)
dim =[100, 140]
dirs = ['AHDBase_TrainingSet','AHDBase_TestingSet']

#download ADBase dataset
for dir in dirs:
    url = f'https://datacenter.aucegypt.edu/shazeem/Files/{dir}.rar'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f'{dir}.rar', 'wb') as f:
            f.write(response.raw.read())
        patoolib.extract_archive(f"{dir}.rar", outdir=".")
    os.remove(f"{dir}.rar")


#make target directory if it doesnt exists and sub directories 
target_dir = 'AHBase'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    for dir in dirs:
        os.makedirs(f"{target_dir}\{dir}")
        for i in range(10):
            if not os.path.exists(f'{target_dir}\{dir}\{i}'):
                os.makedirs(f'{target_dir}\{dir}\{i}')

#read and pad bmp images and save them into digit folder as pngs
for dir in dirs:
    for folder in os.listdir(f'{dir}'):
        i = 0
        for filename in os.listdir(f'{dir}\{folder}'):
            if filename.split(".")[1] == 'bmp':
                img = Image.open(f'{dir}\{folder}\{filename}')
                x = np.array(img)*1
                xdim = (140 - x.shape[1])
                ydim = (100 - x.shape[0])
                xdiv = xdim//2
                ydiv = ydim//2
                xmod = xdim%2
                ymod = ydim%2
                x = np.pad(x,((ydiv, ydiv+ymod) ,(xdiv,xdiv+xmod)), 'constant', constant_values = [1] )
                x = x.astype(np.uint8)*255
                output = Image.fromarray(x)
                newfilename = filename.split('.')[0]
                output.save(f'{target_dir}\{dir}\{filename.split(".")[0][-1]}\{newfilename}.png')