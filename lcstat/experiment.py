import lcstat.config as conf
from lcstat.helpers import *

from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
import math

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
    def __init__(self, dataset, lf, pretty_lf, kernel, c, max_steps, method="lc"):
        self.method = method
        self.dataset = dataset
        self.lf = lf
        self.pretty_lf = pretty_lf
        self.kernel = kernel
        self.c = c
        self.max_steps = max_steps

    def train_model_lc(self, objects, tag):
        train_file = touch_file_with_list(self.dataset, tag, objects)
        model_file = tmp_file()

        res = run([conf.LC_TRAIN, self.lf, self.kernel, str(self.c), str(self.max_steps), train_file, model_file])
        if res.stderr:
            print("out: %s" % res.stdout)
            print("err: %s" % res.stderr)

        return model_file

    def test_model_lc(self, model, objects, tag):
        test_objects = touch_file_with_list(self.dataset, tag, objects)
        res = run([conf.LC_PREDICT, test_objects, model, "stdout"])
        if res.stderr:
            print("out: %s" % res.stdout)
            print("err: %s" % res.stderr)
            print("objects %s" % objects)
            raise Exception("Broken %s" % model)
        else:
            lines = res.stdout.decode().split("\n")
            return [int(x) for x in lines[0].split()], \
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

    def run_on_one_sample(self, index, train, test):
        if self.method == "svm":
            model = self.train_model_svm(train)
            labels, values = self.test_model_svm(model, test)
            return labels, values
        else:
            model = self.train_model_lc(train, "train_%d" % index)
            labels, values = self.test_model_lc(model, test, "test_%d" % index)
            return labels, values

    def perform(self):
        print("Performing %s" % vars(self))
        objects = np.array(tuple(open(self.dataset, 'r')))
        print("Samples %s" % len(objects))
        self.labels = []
        self.values = []
        index = 100 * conf.SPLITS
        for train_index, test_index in KFold(n_splits=conf.SPLITS).split(objects):
            train, test = objects[train_index], objects[test_index]
            l, v = self.run_on_one_sample(index, train, test)
            self.labels += l
            self.values += v
            index += 1

        return self.labels, self.values

    def verify(self):
        for v in self.values:
            if math.isnan(v):
                return False
        return True

    def str(self):
        if self.method == "lc":
            tag = "lf: {0}, c: {1}, step: {2} kernel: {3}".format(self.pretty_lf, self.c, self.max_steps, self.kernel)
        elif self.method == "svm":
            tag = "c: {0} kernel: {1}".format(self.c, self.kernel)
        elif self.method == "bayes":
            tag = ""
        else:
            print("\n\n\n\tError in type")
            raise Exception("Malformed experiment")
        return self.method + " " + tag
