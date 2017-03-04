#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import click
import os

def roc(labels, values, name, output):
    fpr, tpr, _ = roc_curve(labels, values)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
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


@click.command()
@click.option("--inp", default="data.predicted", help="Utils folder")
@click.option("--out", default="roc.png", help="Data folder")
@click.option("--name", default="Unknown", help="Data folder")
def build_roc(inp, out, name):
    lines = []
    with open(inp, "r") as content:
        for line in content:
            lines.append(line)

    labels = [ int(num) for num in lines[0].split()]
    values = [ float(num) for num in lines[1].split()]

    print("Parsed " + inp)
    roc(labels, values, name, out)

if __name__ == '__main__':
    build_roc()