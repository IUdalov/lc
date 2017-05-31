#!/usr/bin/env python3

import math
import argparse
from lcstat.helpers import file_content
import sys
import os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

word_to_strip = set(["the", "a", "then", "in",
                     "into", "в", "into", "near",
                     "over", "is", "was", "has",
                     "an", "etc", "ps", "and", "or", "vs"])

def sanitize(raw_s):
    s = raw_s.lower()
    if s in word_to_strip:
        return ""
    if s.startswith("http"):
        return "(link)"
    else:
        return "".join(
            c for c in s if c.isalpha())

all_words = {}
def register_world(word):
    if not all_words.get(word):
        register_world.counter += 1
        all_words[word] = register_world.counter
register_world.counter = 0

def get_words(path):
    content = file_content(path)
    words = []
    for s in content.split():
        new_s = sanitize(s)
        if new_s != "":
            words.append(new_s)
    count_words = {}
    for word in words:
        count = count_words.get(word, 0)
        count_words[word] = count + 1

    return count_words


parser = argparse.ArgumentParser(description='Makes tf-idf features')
parser.add_argument('-f', metavar='N', nargs='+',
                    help='objects of first class')
parser.add_argument('-s', metavar='N', nargs='+',
                    help='objects of second class')
args = parser.parse_args()

# precess files
path_to_class = {}
for f in args.f:
    path_to_class[f] = 1
for s in args.s:
    path_to_class[s] = -1
#eprint(path_to_class)

path_to_words = {}
bad_files = []
for path, c in path_to_class.items():
    try:
        path_to_words[path] = get_words(path)
        eprint("Parsed: ", path)
    except:
        bad_files.append(path)
        eprint("Skipped: ", path)

for path in bad_files:
    del path_to_class[path]

idf_tmp = {}
for path, words in path_to_words.items():
    for word in words:
        register_world(word)
        count = idf_tmp.get(word, 0)
        idf_tmp[word] = count + 1

ndocs = len(path_to_words)
idf = {}
for word, n in idf_tmp.items():
    idf[word] = math.log(ndocs/n)

path_to_features = {}
for path, words in path_to_words.items():
    tmp = {}
    nwords = 0
    for word, n in words.items():
        nwords += n

    for word, n in words.items():
        tmp[word] = (n / nwords) * idf[word]
    path_to_features[path] = tmp

eprint(all_words)

def features_to_str(features):
    str = ""
    for word, tfidf in features.items():
        str += " {num}:{value}".format(num=all_words[word], value=tfidf)
    return str + " {num}:0".format(num=register_world.counter + 1)

for path, c in path_to_class.items():
    str = "+1" if c == 1 else "-1"
    str += features_to_str(path_to_features[path])
    print(str)

