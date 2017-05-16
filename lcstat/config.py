LFS = {
    "X":    "X",
    "X3_2": "X3/2",
    "X2":   "X2",
#    "X3":   "X3",
#    "X4":   "X4",
    "E":    "E",
    "S":    "S",
    "L":    "L"
}

KERNELS = [
    "H1",
    # "H2",
    #"H3",
    #"I1",
    # "I2",
    #"I3",
    # "RAD",
    # "GRAD",
    #"HYP"
]

APPROX = [
    "Gauss",
    "Poisson",
    "Bernoulli"
    #"Binomial"
]

STEPS = [1, 5, 10, 100]
CONSTANTS = [1]#, 0.1]#, 0.01] #, 0.001]

LC_TRAIN = "bin/lc-train"
LC_PREDICT = "bin/lc-predict"
SPLITS = 10

ROC_DIR = "roc3"
DATA_DIR = "data2"
TMP_DIR = "/tmp"
