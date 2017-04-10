#!/usr/bin/env python3

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np

import os
import subprocess
import sys

# LFS = ["V"]
#STEPS = [0, 10]
#CONSTANTS = [1]
LFS = ["V", "Q", "Q3", "Q4"]
STEPS = [0, 10, 100] #, 1000]
CONSTANTS = [1, 0.1, 0.01, 0.001]

LC_TRAIN = "bin/lc-train"
LC_PREDICT = "bin/lc-predict"
SPLITS = 10

COLORS = ['darkorange', 'cyan', 'indigo', 'seagreen', 'yellow', 'blue', "green", "pink", "grey", "purple", "black"]

ROC_DIR = "roc3"
DATA_DIR = "data2"
TMP_DIR = "tmp"

# utils
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def tmp_file():
    tmp_file.counter += 1
    return TMP_DIR + "/tmp" +  str(tmp_file.counter)
tmp_file.counter = 0

def tmp_file_with_list(ls):
    file_path = tmp_file()
    with open(file_path, "w") as f:
        for l in ls:
            f.write("%s" % l)
    return file_path

def basename(name):
    return name.split("/")[-1].split(".")[0]

# ROC and graphic
def roc(experiments, name, output):
    plt.figure()
    lw = 2
    for index, e in enumerate(experiments):
        fpr, tpr, _ = roc_curve(e.labels, e.values)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=COLORS[index], lw=lw, label= e.str() + ' (area = %0.5f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ' + name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(output, dpi=100)
    print("Saved to " + output)

def grouped_roc(exp, name, predicate):
    try:
        by_steps = dict()
        for e in exp:
            key = predicate(e)
            ls = by_steps.get(key, [])
            ls.append(e)
            by_steps[key] = ls

        for k, exps in by_steps.items():
            roc(exps, k, ROC_DIR + "/" + name + k + ".png")

    except:
        print("\tUnexpected error: ", sys.exc_info()[0], exp)

# data manipulation
def parse_object(object):
    words = object.split()
    c = 1 if words[0] == "+1" else -1
    rest = words[1:]
    ind_to_feature = dict()
    for e in rest:
        [i_str, f_str] = e.split(":")
        ind_to_feature[int(i_str)] = float(f_str)
    nfeatures =  max(ind_to_feature, key=int)
    features = []
    for i in range(1, nfeatures):
        features.append(ind_to_feature.get(i, 0))

    return c, features

class Experiment:
    def __init__(self, dataset, lf, c, max_steps, method="lc"):
        self.method = method
        self.dataset = dataset
        self.lf = lf
        self.c = c
        self.max_steps = max_steps

    def train_model_lc(self, objects):
        train_file = tmp_file_with_list(objects)
        model_file = tmp_file()

        print("Train " + train_file + " -> " + model_file)

        res = run([LC_TRAIN, self.lf, str(self.c), str(self.max_steps), train_file, model_file])
        if res.stderr:
            print("out: %s" % res.stdout)
            print("err: %s" % res.stderr)

        return model_file

    def test_model_lc(self, model, objects):
        test_objects = tmp_file_with_list(objects)
        print("Test " + model + " -> " + test_objects)
        res = run([LC_PREDICT, test_objects, model, "stdout"])
        if res.stderr:
            print("out: %s" % res.stdout)
            print("err: %s" % res.stderr)
            print("objects %s" % objects)
            raise Exception("Broken %s" % model)
        else:
            lines = res.stdout.decode().split("\n")
            return [int(x) for x in lines[0].split()],\
                   [float(x) for x in lines[1].split()]

    def train_model_svm(self, objects):
        print("Train on %d" % len(objects))
        x = []
        y = []
        for object in objects:
            yx, xx = parse_object(object)
            y.append(yx)
            x.append(xx)
        clf = SVC(probability=True, kernel='linear', C=self.c)
        clf.fit(x, y)
        return clf

    def test_model_svm(self, model, objects):
        print("Test on %d" % len(objects))
        labels = []
        to_predict = []
        for o in objects:
            c, features = parse_object(o)
            labels.append(c)
            to_predict.append(features)
        values = model.decision_function(to_predict).tolist()
        return labels, values

    def run_on_one_sample(self, train, test):
        if self.method == "svm":
            model = self.train_model_svm(train)
            labels, values = self.test_model_svm(model, test)
            return labels, values
        else:
            model = self.train_model_lc(train)
            labels, values = self.test_model_lc(model, test)
            return labels, values

    def perform(self):
        print("Performing %s" % vars(self))
        objects = np.array(tuple(open(self.dataset, 'r')))
        print("Samples %s" % len(objects))
        self.labels = []
        self.values = []
        for train_index, test_index in KFold(n_splits=SPLITS).split(objects):
            train, test = objects[train_index], objects[test_index]
            l, v = self.run_on_one_sample(train, test)
            self.labels += l
            self.values += v
        # roc([self], self.id, ROC_DIR + "/" + self.id + ".png")

        return self.labels, self.values

    def str(self):
        return basename(self.dataset) + " " \
                + self.method \
                + " lf: {0}, c: {1}, steps: {2}".format(self.lf, self.c, self.max_steps)

def run_on_dataset(ds):
    mkdir(ROC_DIR)
    mkdir(TMP_DIR)
    all_experiments = []
    for experiment in experiments(ds):
        try:
            experiment.perform()
            all_experiments.append(experiment)
        except ValueError:
            print("\tCould not convert data: ", experiment)
        except:
            print("\tUnexpected error: ", sys.exc_info()[0], experiment)

    dataset_name = basename(ds)
    grouped_roc(all_experiments, "%s_by_ms" % dataset_name, lambda e: "lf{0}_c{1}".format(e.lf, e.c))
    grouped_roc(all_experiments, "%s_by_c" % dataset_name,  lambda e: "lf{0}_ms{1}".format(e.lf, e.max_steps))
    grouped_roc(all_experiments, "%s_by_lf" % dataset_name, lambda e: "c{0}_ms{1}".format(e.c, e.max_steps))


def datasets(data_dir):
    files = os.listdir(data_dir)
    for file in files:
        print("Dataset: %s" % file)
        yield data_dir + "/" + file


def experiments(ds):
    for c in CONSTANTS:
        for ms in STEPS:
            yield Experiment(ds, "V", c, ms, method="svm")
            for lf in LFS:
                yield Experiment(ds, lf, c, ms)

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    for arg in args:
        if os.path.isdir(arg):
            for ds in datasets(arg):
                run_on_dataset(ds)
        else:
            run_on_dataset(arg)
