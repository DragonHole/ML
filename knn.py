import numpy as np
import operator as op

# useful only in the beginning
def createDataSet():
    groups = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels

def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOlines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOlines)
    # 返回的NumPy矩阵numberOfLines行，3列
    returnMat = np.zeros((numberOfLines, 3))
    # 创建分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    # 读取每一行
    for line in arrayOlines:
        # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
        line = line.strip()
        # 将每一行内容根据'\t'符进行切片,本例中一共有4列
        listFromLine = line.split('\t')
        # 将数据的前3列进行提取保存在returnMat矩阵中，也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        # 根据文本内容进行分类1：不喜欢；2：一般；3：喜欢
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    # 返回标签列向量以及特征矩阵
    return returnMat, classLabelVector

# normalize formula: newVal = (oldVal - min) / (max - min)
def normalize(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    normDataSet = np.zeros(np.shape(dataSet))   # prepare an empty matrix
    m = dataSet.shape[0]    # rows(how many training examples)
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(maxVals-minVals, (m, 1))    # element-wise division
    
    return normDataSet, maxVals-minVals, minVals


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]      # number of rows 
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # replicate the inX as a Mx1 matrix where M is the number of examples, and take the difference
    sqDiffMat = diffMat**2  # euclidean distance 
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()

    classCount={}
    for i in range(k):  # take the top kth training set example
        voteILabel = labels[sortedDistIndicies[i]] # see which label(y) it belongs to 
        classCount[voteILabel] = classCount.get(voteILabel, 0)+1    # calculate the frequency of this label 
    
    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)   # from highest to smallest frequency

    # print(sortedClassCount)
    return sortedClassCount[0][0]   # take the most frequent label(y) 


if __name__ == "__main__":
    resultList = ['result 1', 'result 2', 'result 3']
    arg1 = float(input("Time spent on video games: "))
    arg2 = float(input("flier miles earned this year: "))
    arg3 = float(input("ice cream consumed: "))

    dataMatrix, labels = file2matrix('datingTestSet.txt')
    normMat, ranges, minvals = normalize(dataMatrix)
    argsMatrix = np.array([arg1, arg2, arg3])

    result = classify0(argsMatrix, dataMatrix, labels, 3)
    print("result = " + str(result))
    