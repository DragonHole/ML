# bayes classifier with naive assumptions 

import numpy as np

def createDataSet():
    sentences = [['my', 'dog', 'has', 'flea', 'problems'],
                ['stupid', 'ass', 'go', 'to', 'hell'],
                ['stupid', 'ass']]
    classVector = [0, 1, 1]  # 1 for classied as abusive, 0 for not abusive
    return sentences, classVector

# given a list of lists of words(dataset), return a list of unique words appeared in the dataset
def createVocabList(dataSet):
    vocabSet = set([])  # set of list, disallow duplicate but retain order 
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    print('vocab:', list(vocabSet))   # 因为set随机打乱顺序，打印出来才能看明白输出概率
    return list(vocabSet)   # return as a list because later need to be work with a list of 0 and 1, and correspondance requires ordered list

# given a word set from a sentence, 看出现了vocab里哪些已知的词汇
def wordsAppearanceVec(vocabList, inputSet):
    appearanceVec = [0]*len(vocabList) # create a vector consists of zeroes
    for word in inputSet:
        if word in vocabList:
            appearanceVec[vocabList.index(word)] = 1
        else:
            print('Error: The word %s is not in not in vocab' % word)
    return appearanceVec

# the previous function only checks if a word appeared, but not its frequency
# this function is a better version, provides more information
def bagOfWords2Vector(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# trainMatrix: a list of 0 and 1 indicating appearance of each vocab 
# trainClasses: a list of class labels for each sentence
def trainNB0(trainMatrix, trainClasses):
    numTrainExample = len(trainMatrix)  # 有几个sentence
    numWords = len(trainMatrix[0])
    p0Numerator = np.ones(numWords) # use ones instead of zeroes to account for zero multiplicated product
    p1Numerator = np.ones(numWords)
    p0Denominator = 2.0 # for the above's(zero multiplicated product) sake, not sure why
    p1Denominator = 2.0 

    for i in range(numTrainExample):
        if trainClasses[i] == 1:
            p1Numerator += trainMatrix[i]      #  把一行的0和1以element wise方法加上去
            p1Denominator += sum(trainMatrix[i])    # bayes formula denominator is p(w)
        else:
            p0Numerator += trainMatrix[i]
            p0Denominator += sum(trainMatrix[i])
    p1Vect = np.log(p1Numerator/p1Denominator)  # log to address 'underflow'
    p0Vect = np.log(p0Numerator/p0Denominator)

    pAbusive = sum(trainClasses)/float(numTrainExample) # 因为1代表abusive，而1可以做加法，而numTrainExample是一共多少份
    return p0Vect, p1Vect, pAbusive 

def classifyNB(vec2Classify, p0Vector, p1Vector, pClass1):
    p0 = sum(vec2Classify * p0Vector) + np.log(pClass1)
    p1 = sum(vec2Classify * p1Vector) + np.log(pClass1)

    if p1 > p0: # simply compare the probability
        return 1
    else:
        return 0

def textParse(bigString):
    import re 
    listOfTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]

# returns the 30 most frequent words
def calcMostFreq(vocabList, fullText):
    import operator as op
    freqDict = {}
    for vocab in vocabList:
        freqDict[vocab] = fullText.count(vocab)

    # rank from high to low 
    sortedFreqDict = sorted(freqDict.items(), key=op.itemgetter(1), reverse=True)
    return sortedFreqDict[:30]  # returns the top 30

if __name__ == '__main__':
    listOfSentences, classList = createDataSet() # classList是每个句子人工提供对应的标签
    myVocabList = createVocabList(listOfSentences)
    trainMat = []
    for sentence in listOfSentences:
        trainMat.append(wordsAppearanceVec(myVocabList, sentence))

    print('train matrix:', trainMat)
    
    p0v, p1v, pAbusive = trainNB0(np.array(trainMat), np.array(classList))
    print('p0: ', p0v)
    print('p1: ', p1v)
    print('pAbusive: ', pAbusive)

    # test 2
    testEntry = ['kiss', 'my', 'stupid', 'ass']
    thisEntryAppearenceVec = wordsAppearanceVec(myVocabList, testEntry)
    print('classified as', classifyNB(thisEntryAppearenceVec, p0v, p1v, pAbusive))


