"""
Re-export fma_medium in different codecs.
"""

import os
import numpy as np
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool

from loader.global_variables import *

OUTDB_PATH = HOME+"/fma_codec"

split_dict = np.load(SPLIT_PATH, allow_pickle=True).item()
test_list = split_dict['test']

pos_list = glob(os.path.join(POS_DB_PATH, '**/*.mp3'))
neg_list = glob(os.path.join(NEG_DB_PATH, '**/**/*.mp3'))


dict_path = {}
encoders = set()
for s_path in neg_list:
    s_path_split = s_path.split("/")
    s_mp3 = s_path_split[-1]
    encoder = s_path_split[-3]
    encoders.add(encoder)
    dict_path[encoder + "." + s_mp3] = s_path

for s_path in pos_list:
    s_mp3 = s_path.split("/")[-1]
    dict_path["real."+s_mp3] = s_path

encoders = sorted(list(encoders))

print("Split contains", len(test_list), "db found:", len(dict_path), 'items')

##

CODEC = input("\nCodec? [mp3/aac/libopus]\n>> ")
BR = input("\nBitrate? [64]\n>> ")
OUT_DIR = "{}_{}".format(CODEC, BR)

for e_name in encoders + ['real']:
    out_path = os.path.join(OUTDB_PATH, OUT_DIR, e_name)
    os.makedirs(out_path, exist_ok = True)

##


def encode_audio(input_filename, output_filename):
    command = f"ffmpeg -y -hide_banner -i {input_filename} -c:a {CODEC} -b:a {BR}k {output_filename} -loglevel panic"
    os.system(command)
    # print(command)
    return 0


for e_name in encoders + ['real']:
    print("Starting pool for", e_name)
    extension = { "mp3": "mp3", "aac": "aac", "libopus": "opus" }
    path_list = [ ( dict_path[ e_name + "." + s_mp3 ],
                    os.path.join(OUTDB_PATH, OUT_DIR, e_name, "{}.{}".format(s_mp3.split('.')[0], extension[CODEC] ) ) )
                    for s_mp3 in test_list ]
    print("Found", len(path_list), "paths")

    p = Pool()
    p.starmap(encode_audio, path_list)

print("Done.")



