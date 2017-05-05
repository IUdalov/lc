import lcstat.config as conf

import os
import subprocess


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run(cmd):
    print("cmd: " + " ".join(cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)

def tmp_file(seed="roc3_dummy"):
    tmp_file.counter += 1
    return conf.TMP_DIR + "/" + seed +  str(tmp_file.counter)
tmp_file.counter = 0

def basename(name):
    return name.split("/")[-1].split(".")[0]

def touch_file_with_list(dataset, tag, ls):
    path = "/tmp/{0}_{1}".format(basename(dataset), tag)
    if os.path.exists(path):
        return path
    with open(path, "w") as f:
        for l in ls:
            f.write("%s" % l)

    print("Created " + path)
    return path