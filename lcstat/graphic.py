import lcstat.config as conf

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def roc(experiments, name, output):
    plt.figure()
    lw = 2
    for index, e in enumerate(experiments):
        fpr, tpr, _ = roc_curve(e.labels, e.values)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=conf.COLORS[index], lw=lw, label= e.str() + ' (AUC = %0.5f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ' + name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(output, dpi=150)
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
            roc(exps, k, conf.ROC_DIR + "/" + name + k + ".png")

    except:
        print("\tUnexpected error: ", sys.exc_info()[0], exp)

def grouped_by_ms(all, dataset_name):
    print("Grouping by max steps")

    tag = "by_ms"
    index = dict()
    init = None
    for e in all:
        if e.method == "lc":
            key = "lf_{0}_k_{1}_c_{2}".format(e.lf, e.kernel, e.c)
            index.setdefault(key, []).append(e)
        elif e.method == "bayes":
            init = e

    for key, exps in index.items():
        exps.append(init)
        if exps[0].lf == "X":
            for e in all:
                if e.c == exps[0].c and e.method == "svm" and e.kernel == exps[0].kernel:
                    exps.append(e)
        roc(exps, \
            "Grouped by steps {0}".format(key), \
            "{0}/{1}_{2}_{3}.png".format(conf.ROC_DIR, dataset_name,tag, key))

def grouped_by_c(all, dataset_name):
    print("Grouping by C")
    tag = "by_c"
    index = dict()
    init = None
    for e in all:
        if e.method == "lc":
            key = "lf_{0}_k_{1}_ms_{2}".format(e.lf, e.kernel, e.max_steps)
            index.setdefault(key, []).append(e)
        elif e.method == "bayes":
            init = e

    for key, exps in index.items():
        exps.append(init)
        if exps[0].lf == "X":
            for e in all:
                if e.method == "svm" and e.kernel == exps[0].kernel:
                    exps.append(e)
        roc(exps, \
            "Grouped by C {0}".format(key), \
            "{0}/{1}_{2}_{3}.png".format(conf.ROC_DIR, dataset_name, tag, key))


def grouped_by_lf(all, dataset_name):
    print("Grouping by Kernel")
    tag = "by_lf"
    index = dict()
    init = None
    for e in all:
        if e.method == "lc":
            key = "k_{0}_c_{1}_ms_{2}".format(e.kernel, e.c, e.max_steps)
            index.setdefault(key, []).append(e)
        elif e.method == "bayes":
            init = e

    for key, exps in index.items():
        exps.append(init)
        if exps[0].lf == "X":
            for e in all:
                if e.method == "svm" and  e.c == exps[0].c and e.kernel == exps[0].kernel:
                    exps.append(e)
        roc(exps, \
            "Grouped by Loss function {0}".format(key), \
            "{0}/{1}_{2}_{3}.png".format(conf.ROC_DIR, dataset_name, tag, key))

def grouped_by_kernel(all, dataset_name):
    print("Grouping by Kernel")
    tag = "by_kernel"
    index = dict()
    init = None
    for e in all:
        if e.method == "lc":
            key = "lf_{0}_c_{1}_ms_{2}".format(e.lf, e.c, e.max_steps)
            index.setdefault(key, []).append(e)
        elif e.method == "bayes":
            init = e

    for key, exps in index.items():
        exps.append(init)
        if exps[0].lf == "X":
            for e in all:
                if e.method == "svm" and e.c == exps[0].c:
                    exps.append(e)
        roc(exps, \
            "Grouped by Loss function {0}".format(key), \
            "{0}/{1}_{2}_{3}.png".format(conf.ROC_DIR, dataset_name, tag, key))
