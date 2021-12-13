import os, glob, shutil, json
import numpy as np
import scipy.io.wavfile

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def read_json(path):

    with open(path, "r") as json_file:
        dict = json.load(json_file)

    return dict

def save_json(path, dict):

    with open(path, 'w') as json_file:
        json.dump(dict, json_file)
