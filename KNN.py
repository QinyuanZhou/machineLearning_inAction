from numpy import *
from os import listdir
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


#k—近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]#shape[0]返回行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet#tile表示inX在行方向上重复datasize次，在列方向上重复一次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#每行的和
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()#argsort按从小到大顺序排列并输出对应的索引
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]#记录前k个的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1#构造字典get键为voteIlabel时加一
    sortedClassCount = sorted(classCount.items(),
     key=operator.itemgetter(1),reverse=True)#reverse=True是降序排列，按照字典中的第二个元素排序
    return sortedClassCount[0][0]


#从文本文件中解析数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()#得到文件行数
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))#创建返回的numpy矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:          #解析文件数据到列表
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index,:] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat,classLabelVector


#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)#返回每列的最小值和最大值
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10#测试比例
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],6)
        print("the classifier came back with:%d,the real answer is:%d"\
              %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):errorCount += 1.0
    print("the total error rate is:%f"%(errorCount/float(numTestVecs)))


#将图像转换为向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


#手写数字识别系统
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,\
                                     trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real answer is:%d"\
                %(classifierResult,classNumStr))
        if(classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is:%d"%errorCount)
    print("\nthe total error rate is:%f"%(errorCount/float(mTest)))


if __name__=='__main__':
    datingClassTest()




