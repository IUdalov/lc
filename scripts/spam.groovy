#!/usr/bin/env groovy

def isGood(word) {
    word ==~ /[a-z]*/
}

def normalizeWord(word) {
    word.toLowerCase().replaceAll("[,.*+/?!':)(`\"=;-]", "")
}

def source = new File(args[0])
def destination = new File(args[1])
println "Parsing: $source"

def classses = []
def features = []

def lines = source.text.tokenize("\n")
println "Lines: ${lines.size()}"
lines.each { line ->
    def (c, fs) = line.tokenize("\t")
    classses.add(c == "ham" ? 1 : -1)
    features.add (fs.tokenize(" ").collect {normalizeWord it}.findAll {isGood it} as Set)
}

def uniqueWords = [] as Set
features.each { uniqueWords.addAll it }

def numWords = [:]
def idx = 1
uniqueWords.each {
    numWords[it] = idx
    idx++
}

println "Uniq: ${uniqueWords.size()}"

def numFeatures = []
features.each {
    numFeatures.add(it.collect { numWords[it] } as Set)
}

//uniqueWords.each { println it }
// println uniqueWords

println "Writing: $destination"
destination.withWriter { w ->
    for(int i = 0; i < classses.size(); i++) {
        w << "${classses[i] == 1 ? "+1" : "-1"} ${numFeatures[i].collect{"$it:1"}.join(" ")}\n"
    }
}