import math


def StatDataInf(data):
    dataLen = len(data)
    diffDataNum = []
    diffData = []
    other = data
    for i in range(dataLen):
        cnt = 0
        j = i
        if (other[j] != '/'):
            temp = other[i]
            diffData.append(temp)
            while (j < dataLen):
                if (other[j] == temp):
                    cnt += 1
                    other[j] = '/'
                j = j + 1
            diffDataNum.append(cnt)
    return diffData, diffDataNum


def DataEntropy(data, diffData, diffDataNum):
    dataLen = len(data)
    diffDataLen = len(diffDataNum)
    entropyVal = 0
    for i in range(diffDataLen):
        proptyVal = diffDataNum[i] / dataLen
        entropyVal = entropyVal - proptyVal * math.log2(proptyVal)
    return entropyVal


def main():
    data = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    [diffData, diffDataNum] = StatDataInf(data)
    entropyVal = DataEntropy(data, diffData, diffDataNum)
    print(entropyVal)
    data = [1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
    [diffData, diffDataNum] = StatDataInf(data)
    entropyVal = DataEntropy(data, diffData, diffDataNum)
    print(entropyVal)
    data = [1, 2, 3, 4, 2, 1, 2, 4, 3, 2, 3, 4, 1, 1, 1]
    [diffData, diffDataNum] = StatDataInf(data)
    entropyVal = DataEntropy(data, diffData, diffDataNum)
    print(entropyVal)
    data = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    [diffData, diffDataNum] = StatDataInf(data)
    entropyVal = DataEntropy(data, diffData, diffDataNum)
    print(entropyVal)
    data = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 5]
    [diffData, diffDataNum] = StatDataInf(data)
    entropyVal = DataEntropy(data, diffData, diffDataNum)
    print(entropyVal)


if __name__ == '__main__':
    main()
