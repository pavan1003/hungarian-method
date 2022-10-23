from cmath import inf
import numpy as np

# def printMatrix(mat):
#     for i in range(Rows):
#         for j in range(Columns):
#             print(mat[i][j], end=" ")
#         print()

# def inputMatrix():
#     mat=[]
#     for i in range(Rows):
#         # taking row input from the user
#         row = list(map(int, input().split()))[:Columns]
#         # appending the 'row' to the 'matrix'
#         mat.append(row)
#     return mat

# Rows = int(input("Give the number of rows:"))
# Columns = int(input("Give the number of columns:"))
# matrix = inputMatrix()
# matrix = np.reshape(matrix,(Rows, Columns))
# #printMatrix(matrix)


# matrix=[
#     [90, 80, 75, 70],
#     [35, 85, 55, 65],
#     [125, 95, 90, 95],
#     [45, 110, 95, 115],
#     [50, 100, 90, 100],
# ]

# matrix=[
#     [2,9,2,7,1],
#     [6,8,7,6,1],
#     [4,6,5,3,1],
#     [4,2,7,3,1],
#     [5,3,9,5,1]
# ]

# matrix = [
#     [10, 5, 13, 15, 16],
#     [3, 9, 18, 13, 6],
#     [10, 7, 2, 2, 2],
#     [7, 11, 9, 7, 12],
#     [7, 9, 10, 4, 12],
# ]

# matrix = [
#     [0, 7, 0, 4, 0],
#     [4, 6, 5, 3, 0],
#     [2, 4, 3, 0, 0],
#     [2, 0, 5, 0, 0],
#     [3, 1, 7, 2, 0],
# ]

# matrix = [
#     [0, 7, 0, 4, 1],
#     [3, 5, 4, 2, 0],
#     [2, 4, 3, 0, 1],
#     [2, 0, 5, 0, 1],
#     [2, 0, 5, 1, 0],
# ]

# matrix = [
#     [10, 12, 19, 11],
#     [5, 10, 7, 8],
#     [12, 14, 13, 11],
#     [8, 15, 11, 9]
# ]

matrix=[
    [2,9,2,7],
    [6,8,'M',6],
    [4,6,5,3],
    ['M',2,7,3]
]


global Rows, Columns
#Rows = len(matrix)
#Columns = len(matrix[0])


def dummyRowCol(dummyMatrix):
    if (Columns != Rows):
        for i in range(abs(Rows-Columns)):
            if (Rows > Columns):
                dummyMatrix = np.append(
                    dummyMatrix, np.zeros((Rows, 1), dtype=int), axis=1)
                #Columns = Columns + 1
            if (Columns > Rows):
                dummyMatrix = np.append(dummyMatrix, np.zeros(
                    (1, Columns), dtype=int), axis=0)

    return dummyMatrix


def zeroRow(zeroRowMatrix):
    for i in range(Rows):
        zeroRowMatrix[i] = zeroRowMatrix[i] - min(zeroRowMatrix[i])

    return zeroRowMatrix


def zeroColumn(zeroColumnMatrix):
    for i in range(Columns):
        zeroColumnMatrix[:, i] = zeroColumnMatrix[:, i] - \
            min(zeroColumnMatrix[:, i])

    return zeroColumnMatrix


def tickRowCol(tickMatrix):

    unticked_R = []
    ticked_R = []
    ticked_C = []
    # print(ticked_R,ticked_C,unticked_R)
    # print(tickMatrix)
    for i in range(Rows):
        unticked_R.append(i)
        # print(np.count_nonzero(tickMatrix[i]==0))
        if (np.count_nonzero(tickMatrix[i] == 0) == 0):
            for j in range(np.count_nonzero(np.isnan(tickMatrix[i]))):
                ticked_C.append(np.argwhere(
                    np.isnan(tickMatrix[i])).flatten()[j])
            ticked_R.append(i)
            pos_nan = np.argwhere(np.isnan(tickMatrix[i]))
            # print(pos_nan)
            for j in range(len(pos_nan)):
                pos_0 = np.argwhere(tickMatrix[:, pos_nan[j][0]] == 0)
                ticked_R.append(pos_0.flatten()[0])
                if (np.isnan(tickMatrix[pos_0][0][0]).any()):
                    # ticked_R.append(np.argwhere(tickMatrix[:, np.argwhere(np.isnan(tickMatrix[pos_0[0][0]]))[0][0]] == 0)[0][0])
                    ticked_C.append(np.argwhere(
                        np.isnan(tickMatrix[pos_0[0][0]]))[0][0])
    
    ticked_R = np.unique(ticked_R)
    unticked_R = np.unique(
        np.delete(unticked_R, np.where(np.in1d(unticked_R, ticked_R))))
    ticked_C = np.unique(ticked_C)
    # print("Ticks:",ticked_R,ticked_C,unticked_R)
    
    tickMatrix[np.isnan(tickMatrix)] = 0
    minimum_of_cut = inf
    for i in range(Rows):
        for j in range(Columns):
            if i in ticked_R and j not in ticked_C:
                if minimum_of_cut > tickMatrix[i][j]:
                    minimum_of_cut = tickMatrix[i][j]
    for i in range(Rows):
        for j in range(Columns):
            if i in ticked_R and j not in ticked_C:
                tickMatrix[i][j] = tickMatrix[i][j]-minimum_of_cut

    for i in unticked_R:
        for j in ticked_C:
            tickMatrix[i][j] = tickMatrix[i][j]+minimum_of_cut

    count_assignment, tickMatrix = assignMatrix(tickMatrix)

    # print(ticked_R,ticked_C,unticked_R,minimum_of_cut)
    return tickMatrix


def assignMatrix(assignMat):

    assignMat = assignMat.astype('float')

    for i in range(Rows):

        if (np.count_nonzero(assignMat[i] == 0) == 1):
            pos = np.argwhere(assignMat[i] == 0)

            for j in range(Columns):
                if (assignMat[j, pos] == 0):
                    assignMat[j, pos] = np.nan

            assignMat[i, pos] = 0

    for i in range(Columns):

        if (np.count_nonzero(assignMat[:, i] == 0) == 1):
            pos = np.argwhere(assignMat[:, i] == 0)
            for j in range(Rows):
                if (assignMat[pos, j] == 0):
                    assignMat[pos, j] = np.nan
            assignMat[pos, i] = 0

    count_assignment = np.count_nonzero(assignMat == 0)

    assignMatrix.counter += 1

    if (assignMatrix.counter > 20):
        assignMat[np.isnan(assignMat)] = 0
        from munkres import Munkres 
        mun= Munkres()
        mat=mun.compute(assignMat)
        assignMat[assignMat == 0 ] = np.nan
        for i in range(len(mat)):
            assignMat[mat[i][0]][mat[i][1]]=0
        # print(mat)
        # print(assignMat)

    elif (count_assignment != Rows):
        assignMat = tickRowCol(assignMat)
    # print(assignMat)

    return count_assignment, assignMat


def Min_To_Max(maxMatrix):
    maxvalue=(maxMatrix[maxMatrix != np.inf]).max()
    # print("maxvalue: ",maxvalue)
    maxMatrix=maxvalue-maxMatrix
    if(maxMatrix[maxMatrix==-inf].size>0):
        maxMatrix[maxMatrix==-inf]=inf
    return (maxMatrix)


def jobCostCal(jobMatrix, matrixClone):
    matrixClone = np.asarray(matrixClone)
    pos = np.argwhere(jobMatrix == 0)
    # print(matrixClone)
    print("\nAssignment Position:")
    for i in range(Rows):
        print(pos[i],"-->",matrixClone[pos[i][0]][pos[i][1]])
    cost = []
    for i in range(Rows):
        # row = pos[i]
        # print(row)
        # print(matrixClone[pos[i][0]][pos[i][1]])
        cost.append(matrixClone[pos[i][0]][pos[i][1]])
    return sum(cost)


Rows = len(matrix)
Columns = len(matrix[0])
matrix = np.reshape(matrix, (Rows, Columns))

if (type(matrix[0][0]) == np.str_):
    for i in range(Rows):
        for j in range(Columns):
            try:
                matrix[i][j].astype('float')
            except:
                matrix[i][j] = inf

    matrix = matrix.astype('float')

matrix = dummyRowCol(matrix)
Rows = len(matrix)
Columns = len(matrix[0])
# matclone=()
# matClone = matrix
matClone = tuple(map(tuple, matrix))
# matClone = matClone.astype('float')
# flag=1
while True:
    Max_Min = input("1. Maximize \n2. Minimize \nEnter Choice: ")
    print(Max_Min)
    if (Max_Min == '1' or Max_Min == '2'):
        break
    else:
        print("Wrong Choice! please choose 1 or 2\n")
        continue

if Max_Min == '1':
    matrix = Min_To_Max(matrix)
    print("\nMaximize Matrix:\n", matrix)

if (Columns != Rows):
    print("\nMatrix After Adding Dummy Row or Column:\n", matrix)

matrix = zeroRow(matrix)
print("\nMatrix After subtracting minimum of row from each element of that row:\n", matrix)

matrix = zeroColumn(matrix)
print("\nMatrix After subtracting minimum of column from each element of that column:\n", matrix)

assignMatrix.counter = 0

count_assignment, matrix = assignMatrix(matrix)
if (count_assignment != Rows):
    matrix = tickRowCol(matrix)

# matrix=tickRowCol(matrix)

print("\nFinal Solution:\n", matrix)

# print("Clone1:\n",matClone)
jobCost = jobCostCal(matrix, matClone)
print("\nJob cost :\n", jobCost)
