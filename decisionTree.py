from math import log 
import operator as op
# 连numpy都不用，对新手简直太友好了。

def calcShannonEntropy(dataSet):
    num_train_examples = len(dataSet)   # dataSet is a numpy matrix 
    class_counts = {}
    for example in dataSet:
        currentLabel = example[-1]
        if currentLabel not in class_counts.keys(): # 0 if uninitialized
            class_counts[currentLabel] = 0
        class_counts[currentLabel] += 1
        
    shannonEntropy = 0.0
    for aClass in class_counts:
        probability = float(class_counts[aClass]/num_train_examples)
        shannonEntropy -= probability*log(probability, 2)
    return shannonEntropy


# 从dataSet中去除掉选定特征（feature）中某个值（crit_value)后，返回这个新的dataSet
def split_feature(dataSet, feature, crit_value):    # crit_value is the critical value for a selected feature
    retDataSet = []                                 # to be used to extracted from the data set. nominal only, not continuous.
    for featVec in dataSet:
        if featVec[feature] == crit_value:
            reducedFeatVec = featVec[:feature]      # extract those before the specified feature
            reducedFeatVec.extend(featVec[feature+1:])  # extract those after the specified feature, hence getting a feat vec without the specified feature.
            retDataSet.append(reducedFeatVec)
    return retDataSet  

# try the split function on each feature and see which end up with the least entropy
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1     # cuz each entry also has a y value, so minus one. 
    bestEntropy = calcShannonEntropy(dataSet)    # the initial, default value, 如果找不到比这更低的熵就啥也不做
    bestFeature = -1

    for feat in range(numFeatures): # try out each feature
        featList = [example[feat] for example in dataSet]   # 从整个dataset里提一个全是那个特征的列表
        uniqueFeatList = set(featList)  # 相同特征值结果一样，跳过以节省时间
        newEntropy = 0.0
        for uniqueNominalVal in uniqueFeatList:
            subDataSet = split_feature(dataSet, feat, uniqueNominalVal) # 把这个特征值过滤后的子集
            prob = len(subDataSet)/(len(dataSet))  # 选到这个子集的概率
            newEntropy += prob*calcShannonEntropy(subDataSet) # 把所有加起来就是entropy
        infogain = bestEntropy - newEntropy     # 这次loop算出来的熵跟最好的熵差多少, 熵越小越好
        if infogain > 0:
            bestEntropy = newEntropy
            bestFeature = feat
    return bestFeature
        

def createTreeBranches(dataSet):
    if 


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'], 
               [0, 1, 'no'], 
               [0, 1, 'no'], 
               [0, 1, 'yes'], 
               [0, 1, 'yes`']]
    classes = ['eat shit', 'drink pee']
    return dataSet, classes


if __name__ == '__main__':
    x, y = createDataSet()
    print(calcShannonEntropy(x))