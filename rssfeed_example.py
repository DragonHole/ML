from numpy import array
import bayes

# analyse rss feed
def localWords(feed1, feed0):
    import feedparser

    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = bayes.textParse(feed1['entries'][i]['summary'])  # we're gonna use the summary of each entry
        docList.append(wordList)    # a list of ALL entry summaries from feed 0 & 1
        fullText.extend(wordList)   # concatenates the existing list with the new list, has duplicates 
        classList.append(1)

        wordList = bayes.textParse(feed0['entries'][i]['summary'])  # we're gonna use the summary of each entry
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # remove the 30 most frequent occurring words, to increase accuracy, 
    # because the frequent words are redundency and structural glue of the language.
    vocabList = bayes.createVocabList(docList)
    top30Words = bayes.calcMostFreq(vocabList, fullText)    # a map {String key, int count}
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0]) 

    import random
    trainingSet = fullText
    testSet = []
    for i in range(int(len(trainingSet)/4)):     # randomly pick 20 samples and use as testing set
        randomIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])
    
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2Vector(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    p0v, p1v, pSpam = bayes.trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2Vector(vocabList, docList[docIndex])
        if bayes.classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    
    print('error rate is: ', float(errorCount/len(testSet)))
    return vocabList, p0v, p1v


if __name__ == '__main__':
    import feedparser
    bbcAsia = feedparser.parse('http://feeds.bbci.co.uk/news/world/asia/rss.xml')
    bbcAfrica = feedparser.parse('http://feeds.bbci.co.uk/news/world/africa/rss.xml')

    vocabList, pAsia, pAfri = localWords(bbcAsia, bbcAfrica)
