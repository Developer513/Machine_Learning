import numpy as np
import random
import math
# 인터프리터가 제공하는 변수와 함수를 직접제어하는 모듈 
import sys

def loadFile(df):
    resultList = []
    f = open(df, 'r')
    for line in f:
        line = line.rstrip('\n')                    # 1행 씩 카운트 
        sVals = line.split(',')                     # "," 로 구분 
        fVals = list(map(np.float32, sVals))        # 데이터셋 에서 피처를 실수형으로 맵핑
        resultList.append(fVals)                    # 결과리스트에 추가 
    f.close()
    return np.asarray(resultList, dtype = np.float32) 

def showVector(v, dec):
    fmt = "%"+str(dec) + "f"
    for i in range(len(v)):
        x = v[i]
        if x >= 0.0: print(' ', end = '')
        print(fmt % x + '  ', end = '')
    print('')

def showMatrix(m, dec):
    fmt = "%." + str(dec)+"f" 
    for i in range(len(m)):
        for j in range(len(m[i])):
            x = m[i,j]
            if x>= 0.0: print(' ', end = '')
            print(fmt % x + ' ', end = '')
        print('')

def showMatrixPartial(m, numRows, dec, indices):
    fmt = "%." + str(dec) + "f"
    lastRow = len(m) -1
    width = len(str(lastRow))
    for i in range(numRows):
        if indices == True:
            print("[", end ='')
            print(str(i).rjust(width), end = '')
            print("] ",end = '')
        for j in range(len(m[i])):
            x = m[i,j]
            if x >= 0.0: print(' ', end = '')
            print(fmt % x + '  ', end = '')
        print('')
    print(" . . . ")
    if indices == True:
        print("[", end='')
        print(str(lastRow).rjust(width), end = '')
        print("] ", end = '')
    for j in range(len(m[lastRow])):
        x = m[lastRow,j]
        if x>= 0.0: print (' ', end = ' ')
        print(fmt % x + '  ',end = '')
    print('')

class NeuralNetwork:
    def __init__(self, numInput, numHidden, numOutput, seed):
        self.ni = numInput
        self.nh = numHidden
        self.no = numOutput

        self.iNodes = np.zeros(shape = [self.ni], dtype = np.float32)
        self.hNodes = np.zeros(shape = [self.nh], dtype = np.float32)
        self.oNodes = np.zeros(shape = [self.no], dtype = np.float32)

        self.ihWeights = np.zeros(shape = [self.ni,self.nh], dtype=np.float32)
        self.hoWeights = np.zeros(shape = [self.nh,self.no], dtype=np.float32)

        self.hBiases = np.zeros(shape = [self.nh], dtype=np.float32)
        self.oBiases = np.zeros(shape = [self.no], dtype=np.float32)

        self.rnd = random.Random(seed)
        self.initializeWeights()
    
    def setWeights(self, weights):
        if len(weights) != self.totalWeights(self.ni, self.nh, self.no):
            print("Warning: lend(weights) error i setWeights()")
        idx = 0
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i,j] = weights[idx]
                idx +=1
        for j in range(self.nh):
            self.hBiases[j] = weights[idx]
            idx +=1
        for j in range(self.nh):
            for k in range(self.no):
                self.oBiases[k] = weights[idx]
                idx +=1
        for k in range(self.no):
                self.oBiases[k] = weights[idx]
                idx +=1
    def getWeights(self):
        tw = self.totalWeights(self.ni, self.nh, self.no)
        result = np.zeros(shape = [tw], dtype=np.float32)
        idx = 0
        for i in range(self.ni):
            for j in range(self.nh):
                result[idx] = self.ihWeights[i,j]
                idx += 1
        for j in range(self.nh):
            result[idx] = self.hBiases[j]
            idx += 1
        for j in range(self.nh):
            for k in range(self.no):
                result[idx] = self.hoWeights[j,k]
                idx += 1
        for k in range(self.no):
            result[idx] = self.oBiases[k]
            idx += 1
        return result
    
    def initializeWeights(self):
        numWts = self.totalWeights(self.ni, self.nh, self.no)
        wts = np.zeros(shape = [numWts], dtype=np.float32)
        lo = -0.01; hi = 0.01
        for idx in range(len(wts)):
            wts[idx] = (hi - lo) * self.rnd.random() + lo
        self.setWeights(wts)

    def computeOutputs(self, xValues):
        hSums = np.zeros(shape = [self.nh], dtype= np.float32)
        oSums = np.zeros(shape = [self.no], dtype= np.float32)
        for i in range(self.ni):
            self.iNodes[i] = xValues[i]
        for j in range(self.nh):
            for i in range(self.ni):
                hSums[j] += self.iNodes[i] * self.ihWeights[i,j]
        for j in range(self.nh):
            hSums[j] += self.hBiases[j]
        for j in range(self.nh):
            self.hNodes[j] = self.hypertan(hSums[j])
        for k in range(self.no):
            for j in range(self.nh):
                oSums[k] += self.hNodes[j] * self.hoWeights[j,k]
        for k in range(self.no):
            oSums[k] += self.oBiases[k]
        softOut = self.softmax(oSums)
        for k in range(self.no):
            self.oNodes[k] = softOut[k]
        result = np.zeros(shape = self.no, dtype=np.float32)
        for k in range(self.no):
            result[k] = self.oNodes[k]
        return result
    def train(self, trainData, maxEpochs, learnRate):
        hoGrads = np.zeros(shape = [self.nh, self.no],dtype = np.float32)
        obGrads = np.zeros(shape = [self.no], dtype = np.float32)
        ihGrads = np.zeros(shape = [self.ni, self.nh], dtype= np.float32)
        hbGrads = np.zeros(shape = [self.nh], dtype= np.float32)
        oSignals = np.zeros(shape = [self.no], dtype= np.float32)
        hSignals = np.zeros(shape = [self.nh], dtype= np.float32)

        epoch = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)        
        numTrainItems = len(trainData)#150
        indices = np.arange(numTrainItems)#

        while epoch < maxEpochs:
            self.rnd.shuffle(indices)# 0~ 149 까지 150개 셔플 
            for ii in range(numTrainItems):# 0~ 149인덱스
                idx = indices[ii]

                for j in range(self.ni):
                    x_values[j] = trainData[idx, j]
                for j in range(self.no): # 0~2
                   # print(indices)
                    t_values[j] = trainData[idx, j+self.ni]
                    # 데이터셋에서 무작위의 행을 선택하고 그 행의 

                self.computeOutputs(x_values)

                for k in range(self.no):
                    derivative = (1-self.oNodes[k]) * self.oNodes[k]
                    oSignals[k] = derivative * (self.oNodes[k] - t_values[k])

                for j in range(self.nh):
                    for k in range(self.no):
                        hoGrads[j,k] = oSignals[k] * self.hNodes[j]

                for k in range(self.no):
                    obGrads[k] = oSignals[k] * 1.0

                for j in range(self.nh):
                    sum = 0.0
                    for k in range(self.no):
                        sum+=oSignals[k] * self.hoWeights[j,k]
                    derivative = (1-self.hNodes[j]) * (1+self.hNodes[j])
                    hSignals[j] = derivative * sum

                for i in range(self.ni):
                    for j in range(self.nh):
                        ihGrads[i,j] = hSignals[j] * self.iNodes[i]

                for j in range(self.nh):
                    hbGrads[j] = hSignals[j] * 1.0

                for i in range(self.ni):
                    for j in range(self.nh):
                        delta = -1.0 *learnRate * ihGrads[i,j]
                        self.ihWeights[i,j] += delta

                for j in range(self.nh):
                    delta = -1.0*learnRate*hbGrads[j]
                    self.hBiases[j] += delta

                for j in range(self.nh):
                    for k in range(self.no):
                        delta = -1.0 * learnRate*hoGrads[j,k]
                        self.hoWeights[j,k] += delta

                for k in range(self.no):
                    delta = -1.0*learnRate * obGrads[k]
                    self.oBiases[k] += delta
            epoch +=1
            if epoch % 10 == 0:
                mse = self.meanSquaredError(trainData)
                print("epoch = " + str(epoch)+ " ms error = %0.4f" %mse)

        result = self.getWeights()
        return result 
    def accuracy(self, tdata):
        num_correct = 0; num_wrong = 0
        x_values = np.zeros(shape = [self.ni],dtype = np.float32)
        t_values = np.zeros(shape = [self.no],dtype = np.float32)

        for i in range(len(tdata)):
            for j in range(self.ni):
                x_values[j] = tdata[i,j]
            for j in range(self.no):
                t_values[j] = tdata[i, j+self.ni]
            y_values = self.computeOutputs(x_values)
            max_index = np.argmax(y_values)
            if abs(t_values[max_index] - 1.0) < 1.0e-5:
                num_correct +=1
            else:
                num_wrong +=1
        return (num_correct *1.0) / (num_correct + num_wrong)
    # 오차제곱합
    def meanSquaredError(self, tdata):
        sumSquaredError = 0.0
        x_values = np.zeros(shape = [self.ni], dtype=np.float32)
        t_values = np.zeros(shape = [self.no], dtype = np.float32)
        for ii in range(len(tdata)):
            for jj in range(self.ni):
                x_values[jj] = tdata[ii,jj]
            for jj in range(self.no):
                t_values[jj] = tdata[ii,jj+self.ni]
            y_values = self.computeOutputs(x_values)
            for j in range(self.no):
                err = t_values[j] - y_values[j]
                sumSquaredError += err * err 
        return sumSquaredError / len(tdata)
    
    @staticmethod
    def hypertan(x):
        if x < -20.0:
            return -1.0
        elif x > 20.0:
            return 1.0
        else:
            return math.tanh(x)

    @staticmethod
    def softmax(oSums):
        result = np.zeros(shape=[len(oSums)], dtype = np.float32)
        m = max(oSums)
        divisor = 0.0
        for k in range(len(oSums)):
            divisor += math.exp(oSums[k]-m)
        for k in range(len(result)):
            result[k] = math.exp(oSums[k]-m) / divisor
        return result
    
    @staticmethod
    def totalWeights(nInput, nHidden, nOuput):
        tw = (nInput*nHidden) + (nHidden * nOuput) + nHidden + nOuput
        return tw

def main():
    numInput = 4
    numHidden = 5
    numOutput = 3
    print("\n Creating a %d-%d-%d neural network " % (numInput, numHidden, numOutput))
    nn = NeuralNetwork(numInput,numHidden,numOutput, seed = 3)
    print("\n loading iris training and test data ")
    trainDataPath = "/home/JS/Machine_Learning/4.Neural_network/iris_train.txt"
    trainDataMatrix = loadFile(trainDataPath)
    showMatrixPartial(trainDataMatrix,4,1,True)
    testDataPath = "/home/JS/Machine_Learning/4.Neural_network/iris_test.txt"
    testDataMatrix = loadFile(testDataPath)
    maxEpochs = 50
    learnRate = 0.05
    print("\nSetting maxEpochs = " +str(maxEpochs))
    print("Setting learning rate %0.3f" % learnRate)
    print("\nStarting training")
    nn.train(trainDataMatrix, maxEpochs, learnRate)
    print("Training complete")
    accTrain = nn.accuracy(trainDataMatrix)
    accTest = nn.accuracy(testDataMatrix)

    print("\nAccuracy on 120-item train data = %0.4f " % accTrain)
    print("Accuracy on 3-iten test data = %0.4f"% accTest)
    print("\nEnd demo\n")

if __name__ == "__main__":
    main()
