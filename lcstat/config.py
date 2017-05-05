LFS = {
    #"X1_5": "(1 - x)^1/2",
    "X":    "1 - x",
    "X3_2": "(1 - x)^3/2",
    "X2":   "(1 - x)^2",
    # "X3":   "(1 - x)^3",
    # "X4":   "(1 - x)^4",
    #"E":    "exp(-x)",
    "S":    "2 * (1  + e^m)^-1",
    #"L":    "log2(1 + e^-m)"
}

KERNELS = [
    "H1",
    # "H2",
    "H3",
    "I1",
    # "I2",
    "I3",
    # "RAD",
    # "GRAD",
     "HYP"
]

STEPS = [5, 10, 100]
CONSTANTS = [1, 0.1]#, 0.01] #, 0.001]

LC_TRAIN = "bin/lc-train"
LC_PREDICT = "bin/lc-predict"
SPLITS = 10

ROC_DIR = "roc3"
DATA_DIR = "data2"
TMP_DIR = "/tmp"
