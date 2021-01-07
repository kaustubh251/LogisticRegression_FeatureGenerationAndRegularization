import numpy as npy
file = open("ex2data2.txt",'rt')
data = file.read()
data1 = data.split()
x = []
y = []
for element in data1:
    ele = element.split(',')
    datapoint = [1]
    count = 0
    for i in range(len(ele)-1):
        datapoint.append(float(ele[i]))
    y.append(int(ele[len(ele)-1]))
    x1 = datapoint[1]
    x2 = datapoint[2]
    k = 2
    while k<=6:
        for j in range(k+1):
            datapoint.append(npy.power(x1, j)*npy.power(x2, k-j))
        k += 1
    x.insert(count, datapoint)
    count += 1

def z(theta, x1):
    Z = 0
    for i in range(len(x1)):
        Z += theta[i]*x1[i]
    return Z

def h(Z):
    return 1/(1 + npy.exp(-Z))

def cost(theta, x, y):
    cost = 0
    for i in range(len(x)):
        x1 = x[i][:]
        Z = z(theta, x1)
        H = h(Z)
        cost += -(y[i]*npy.log(H) + ((1-y[i])*npy.log(1-H)))
    return cost/len(x)

def gradDescent(theta, x, y, alpha, maxIter, regParam):
    for k in range(maxIter):
        theta1 = theta
        for i in range(len(theta)):
            derCost = 0
            j = 0
            for j in range(len(x)):
                x1 = x[j][:]
                Z = z(theta, x1)
                H = h(Z)
                derCost += (H - y[j])*x[j][i] + regParam*theta[i]
            theta1[i] = theta[i] - alpha*derCost/len(x)      
        theta = theta1
    return theta

def accuracy(theta_Final, x, y):
    q = 0
    p = []
    for i in range(len(x)):
        x1 = x[i][:]
        Z = z(theta_Final, x1)
        H = h(Z)
        if H>=0.5:
            p.insert(i, 1)
        if H<0.5:
            p.insert(i, 0)
        if y[i]==p[i]:
            q += 1
    accuracy = q*100/len(y)
    return accuracy

init_Theta = []
for i in range(len(x[0])):
    init_Theta.append(1)
print(init_Theta)
print("Enter 2 datapoints:")
init_Datapoint = [1]
init_Datapoint.append(float(input()))
init_Datapoint.append(float(input()))
x1 = init_Datapoint[1]
x2 = init_Datapoint[2]
k = 2
while k<=6:
    for j in range(k+1):
        init_Datapoint.append(npy.power(x1, j)*npy.power(x2, k-j))
    k += 1
print("Enter learning rate:")
alpha = float(input())
print("Enter maximum number of iterations for Gradient Descent to converge:")
maxIter = int(input())
regParam = 1000
theta_Final = gradDescent(init_Theta, x, y, alpha, maxIter, regParam)
Z = z(theta_Final, init_Datapoint)
H = h(Z)
print("The probability of microchip to be tested okay is:")
print(H)
print("The accuracy of the learning algorithm is:")
print(accuracy(theta_Final, x, y))
