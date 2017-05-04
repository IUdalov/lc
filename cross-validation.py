#!/usr/bin/env python3

import lcstat.config as conf
from lcstat.helpers import *
import lcstat.graphic as roc
from lcstat.experiment import Experiment
import sys

def run_on_dataset(ds):
    mkdir(conf.ROC_DIR)
    mkdir(conf.TMP_DIR)
    all_experiments = []
    junk_experiments = []
    for experiment in experiments(ds):
        try:
            experiment.perform()
            if experiment.verify():
                all_experiments.append(experiment)
            else:
                junk_experiments.append(experiment)

        except ValueError:
            print("\tCould not convert data: ", experiment)
        except:
            junk_experiments.append(experiment)
            print("\tUnexpected error: ", sys.exc_info()[0], experiment)

    dataset_name = basename(ds)
    roc.grouped_by_ms(all_experiments, dataset_name)
    roc.grouped_by_c(all_experiments, dataset_name)
    roc.grouped_by_lf(all_experiments, dataset_name)
    roc.grouped_by_kernel(all_experiments, dataset_name)

    for e in junk_experiments:
        print("BROKEN {0}".format(e.str()))


def datasets(data_dir):
    files = os.listdir(data_dir)
    for file in files:
        print("Dataset: %s" % file)
        yield data_dir + "/" + file


def experiments(ds):
    yield Experiment(ds, "X", "no pretty", "H1", 1, 0, method="bayes")

    for c in conf.CONSTANTS:
        yield Experiment(ds, "X", "no pretty", "H1", c, 0, method="svm")

    for c in conf.CONSTANTS:
        for ms in conf.STEPS:
            for kernel in conf.KERNELS:
                for lf, pretty_lf in conf.LFS.items():
                    yield Experiment(ds, lf, pretty_lf, kernel, c, ms)
def print_usage():
    print("Usage: {script} [datasets]".format(script=os.path.basename(__file__)))
    print("\tnote: datasets encoded in SVM format")

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print_usage()
        sys.exit(1)
    for arg in args:
        if os.path.isdir(arg):
            for ds in datasets(arg):
                run_on_dataset(ds)
        else:
            run_on_dataset(arg)
