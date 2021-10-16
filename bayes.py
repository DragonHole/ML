# naive bayes with assumptions 

import numpy as np

def createDataSet():
    sentences = [['my', 'dog', 'has', 'flea', 'problems'],
                ['stupid', 'ass', 'go', 'to', 'hell'],
                ['stupid', 'ass']]
    classVector = [0, 1, 1]  # 1 for classied as abusive, 0 for not abusive
    return sentences, classVector

def createVocabList(dataSet):
    vocabSet = set([])  # set of list, disallow duplicate but retain order 
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    print(list(vocabSet))   # 因为set随机打乱顺序，打印出来才能看明白输出概率
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

# trainMatrix: a list of 0 and 1 indicating appearance of each vocab 
# trainClasses: a list of class labels for each sentence
def trainNB0(trainMatrix, trainClasses):
    numTrainExample = len(trainMatrix)  # 有几个sentence
    numWords = len(trainMatrix[0])
    p0Numerator = np.zeros(numWords)
    p1Numerator = np.zeros(numWords)
    p0Denominator = 0.0
    p1Denominator = 0.0

    for i in range(numTrainExample):
        if trainClasses[i] == 1:
            p1Numerator += trainMatrix[i]      #  把一行的0和1以element wise方法加上去
            p1Denominator += sum(trainMatrix[i])    # bayes formula denominator is p(w)
        else:
            p0Numerator += trainMatrix[i]
            p0Denominator += sum(trainMatrix[i])
    p1Vect = p1Numerator/p1Denominator
    p0Vect = p0Numerator/p0Denominator

    pAbusive = sum(trainClasses)/float(numTrainExample) # 因为1代表abusive，而1可以做加法，而numTrainExample是一共多少份
    return p0Vect, p1Vect, pAbusive 


if __name__ == '__main__':
    listOfSentences, classList = createDataSet() # classList是每个句子人工提供对应的标签
    myVocabList = createVocabList(listOfSentences)
    trainMat = []
    for sentence in listOfSentences:
        trainMat.append(wordsAppearanceVec(myVocabList, sentence))
    
    p0v, p1v, pAbusive = trainNB0(trainMat, classList)
    print('p0: ', p0v)
    print('p1: ', p1v)
    print('pAbusive: ', pAbusive)