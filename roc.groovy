#!/usr/bin/env groovy

def buildDir = new File("roc")
def dataDir = new File("data2")
def rocBuilder = "scripts/roc2.py"
def binDir = new File("bin")
def verbose = false
buildDir.mkdirs()

def exec(cmd) {
    def sout = new StringBuilder(), serr = new StringBuilder()
    def proc = cmd.execute()
    proc.consumeProcessOutput(sout, serr)
    proc.waitFor()
    return [sout, serr]
}

def splitFiles = { target ->
    def problem = target.text.tokenize("\n")
    def parts = problem.collate((int)(problem.size() / 10))
    def trainToTest = [:]
    10.times { num ->
        def trainFile = new File("$buildDir/${target.name}.${num}.train")
        trainFile.withWriter { w ->
            parts[num].each { e ->
                w << e << "\n"
            }
        }
        def testFile = new File("$buildDir/${target.name}.${num}.test")
        testFile.withWriter { w ->
            parts.findAll{ it != parts[num] }.each { p ->
                p.each { e ->
                    w << e << "\n"
                }
            }
        }
        trainToTest[trainFile] = testFile
    }
    return trainToTest
}

def train = { exp, trainFile, modelFile ->
    println "train -> $trainFile -> $modelFile"
    def (out, err) = exec("$binDir/lc-train $exp.lf $exp.c $exp.maxSteps $trainFile $modelFile")
    if (err || verbose) {
        println "out: \n$out"
        println "err: \n$err"
    }
}


def test = { testFile, modelFile, predictedFile ->
    println "test -> $testFile -> $modelFile -> $predictedFile"
    def (out, err) = exec("$binDir/lc-predict $testFile $modelFile $predictedFile")
    if (err || verbose) {
        println "out: \n$out"
        println "err: \n$err"
    }
}

def buildRoc = { predictedFile, rocFile ->
    println "roc -> $predictedFile -> $rocFile"
    (out, err) = exec("$rocBuilder --inp $predictedFile --out $rocFile --name ${rocFile.name.tokenize(".")[0]}")
    if (err || verbose) {
        println "out: \n$out"
        println "err: \n$err"
    }
}

def targets = dataDir.listFiles().findAll { it.name.endsWith(".data") }
def trainToTest = targets.collect { splitFiles(it) }


class Experiment {
    def lf;
    def c;
    def maxSteps;
    def dataset; // map: trainFile -> testFile
}

def experiments = []
["V", "Q", /*"Q3", "Q4", "L", "S", "E"*/].each { lf ->
    [1, 0.1, 0.01, 0.001/*, 0.0001*/].each { c ->
        [0, 10, 100/*, 1000*/].each { steps ->
            trainToTest.each { dataset ->
                def e = new Experiment()
                e.lf = lf
                e.c = c
                e.maxSteps = steps
                e.dataset = dataset
                experiments << e
            }
        }
    }
}

experiments.each { exp ->
    def tag = ".exp_l${exp.lf}_c${exp.c}_s${exp.maxSteps}"
    def labels = ""
    def values = ""
    def lastBasename = null

    println "TAG: $tag"
    exp.dataset.each { trainFile, testFile ->
        def basename = trainFile.path - ".train" + tag
        lastBasename = basename

        def modelFile = new File("${basename}.model")
        train(exp ,trainFile, modelFile)

        def predictedFile = new File("${basename}.predicted")
        test(testFile, modelFile, predictedFile)

        def (l, v) = predictedFile.text.tokenize("\n")
        labels += l
        values += v

        //buildRoc(predictedFile, new File("${basename}.png"))
    }

    def base = "${lastBasename.tokenize(".").dropRight(2).join(".")}.${tag}.FINAL"
    def rocFile = new File("${base}.png")
    def predictedFile = new File("${base}.predicted")
    predictedFile.withWriter { w ->
        w << labels << "\n"
        w << values << "\n"
    }

    buildRoc(predictedFile, rocFile)
}

