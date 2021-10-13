import numpy as np
import random
import statistics
from statistics import variance
import matplotlib.pyplot as plt
# first we generate the data set
v = np.random.normal(0,1,100000)
data = [v[0],v[1]]
for i in range(2,100000):
    data.append(v[i] - (0.5*v[i-1]*v[i-2]))

#Now we implement an MLP with 6 input nodes, 16 hidden neurons and 1 output neuron
# 'a' is a 7X16 matrix(including bias) and 'b' is a 17X1 matrix(including bias)

a = np.random.rand(7,16)
b = np.random.rand(17,1)
a_list = [a,a]
b_list = [b,b]
sample_range = []
count = 0
var2 = []
error = []
for i in range(5,99000):
    sample_range.append(i)
for epoch in range(2500):
    start = random.sample(sample_range,1)
    #print(start)
    for i in range(start[0], start[0] + 1000):
        count = count + 1
        temp_inp = np.array([1,v[i],v[i-1],v[i-2],v[i-3],v[i-4],v[i-5]])#1X7 vector
        #this is for the forward propagation
        hidden_output = np.dot(temp_inp,a)#1X16 vector
        hidden_activation = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype = 'float')
        for j in range(1,17):
            hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X17 vector(first element is the bias), these serve as inputs for the output layer
        output = np.dot(hidden_activation,b)#this is a 1X1 matrix
        output = output[0]
        #print(output)
        #final_output = 1/(1 + np.exp(-output)) #1X1 vector, this is the final_output
        #this completes the forward propagation, now we implement the backward propagation
        des = v[i]
        #first we compute the local gradient of the output nodes
        output_loc_grad = (des - output)#1X1 vector
        #print(output_loc_grad)
        #now to calculate the local gradient of the neurons in the hidden layer
        bT = np.transpose(b)
        #temp_act = np.array([hidden_activation[1],hidden_activation[2]])#excludes the bias from the neurons in the hidden layer
        hidden_loc_grad = np.multiply(np.multiply(output_loc_grad*bT,hidden_activation),(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) - hidden_activation))#1X17 vector
        hidden_loc_grad1 = np.array([hidden_loc_grad[0][1],hidden_loc_grad[0][2],hidden_loc_grad[0][3],hidden_loc_grad[0][4],hidden_loc_grad[0][5],hidden_loc_grad[0][6],hidden_loc_grad[0][7],hidden_loc_grad[0][8],hidden_loc_grad[0][9],hidden_loc_grad[0][10],hidden_loc_grad[0][11],hidden_loc_grad[0][12],hidden_loc_grad[0][13],hidden_loc_grad[0][14],hidden_loc_grad[0][15],hidden_loc_grad[0][16]])#this is a 1X16 vector which removes the bias neuron
        #print(hidden_loc_grad1)
        # now to update the weights after every Iterations
        learn = 0.001
        alpha = 0.6
        #first we update the weights matrix 'a'
        for r in range(7):
            for c in range(16):
                a[r][c] = a[r][c] + ((a_list[count][r][c] - a_list[count-1][r][c])*alpha) + learn*hidden_loc_grad1[c]*temp_inp[r]# includes the momentum term

        # now we update the weights matrix 'b'
        for r in range(17):
            for c in range(1):
                b[r][c] = b[r][c] + ((b_list[count][r][c] - b_list[count-1][r][c])*alpha) + learn*output_loc_grad*hidden_activation[r]#includes the momentum term
        a_list.append(a)
        b_list.append(b)
    # compute variance plot and plot Error
    #print("Matrix A: ",a)
    #print("Matrix B:", b)
    if epoch % 50 == 0:
        print(epoch)
        var1 = []
        var_calc = np.random.normal(0,1,1000)
        mse = 0
        for j in range(5,1000):
            temp_inp = np.array([1,var_calc[j],var_calc[j-1],var_calc[j-2],var_calc[j-3],var_calc[j-4],var_calc[j-5]])#1X7 vector
            #this is for the forward propagation
            hidden_output = np.dot(temp_inp,a)#1X16 vector
            hidden_activation = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype = 'float')
            for k in range(1,17):
                hidden_activation[k] = 1/(1 + np.exp(-hidden_output[k-1]))#1X17 vector(first element is the bias), these serve as inputs for the output layer
            output = np.dot(hidden_activation,b)#this is a 1X1 matrix
            output = output[0]
            des = var_calc[j] + (0.5*var_calc[j-1]*var_calc[j-2])
            mse = mse + (output - des)*(output - des)
            #final_output = 1/(1 + np.exp(-output)) #1X1 vector, this is the final_output
            var1.append(output)
        #print(var1)
        var = variance(var1)
        #print(var)
        var2.append(var)
        error.append(mse)
#plotting
x_list = []
for i in range(1, 2501):
    if i%50 == 0:
       x_list.append(i)
plt.scatter(x_list,var2)
plt.plot(x_list,var2)
plt.xlabel("Epochs")
plt.ylabel("Variance")
plt.title("Variance vs Epochs")
plt.show()
plt.scatter(x_list,error)
plt.plot(x_list,error)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Error Trajectory")
plt.show()

'''

# this is for the cross validation
a = np.random.rand(7,16)
b = np.random.rand(17,1)
a_list = [a,a]
b_list = [b,b]
sample_range = []
count = 0
error = []
for i in range(5,98000):
    sample_range.append(i)
for epoch in range(1000):
    start = random.sample(sample_range,1)
    #print(start)
    for i in range(start[0], start[0] + 1000):
        count = count + 1
        temp_inp = np.array([1,v[i],v[i-1],v[i-2],v[i-3],v[i-4],v[i-5]])#1X7 vector
        #this is for the forward propagation
        hidden_output = np.dot(temp_inp,a)#1X16 vector
        hidden_activation = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype = 'float')
        for j in range(1,17):
            hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X17 vector(first element is the bias), these serve as inputs for the output layer
        output = np.dot(hidden_activation,b)#this is a 1X1 matrix
        output = output[0]
        #print(output)
        #final_output = 1/(1 + np.exp(-output)) #1X1 vector, this is the final_output
        #this completes the forward propagation, now we implement the backward propagation
        des = v[i]
        #first we compute the local gradient of the output nodes
        output_loc_grad = (des - output)#1X1 vector
        #print(output_loc_grad)
        #now to calculate the local gradient of the neurons in the hidden layer
        bT = np.transpose(b)
        #temp_act = np.array([hidden_activation[1],hidden_activation[2]])#excludes the bias from the neurons in the hidden layer
        hidden_loc_grad = np.multiply(np.multiply(output_loc_grad*bT,hidden_activation),(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) - hidden_activation))#1X17 vector
        hidden_loc_grad1 = np.array([hidden_loc_grad[0][1],hidden_loc_grad[0][2],hidden_loc_grad[0][3],hidden_loc_grad[0][4],hidden_loc_grad[0][5],hidden_loc_grad[0][6],hidden_loc_grad[0][7],hidden_loc_grad[0][8],hidden_loc_grad[0][9],hidden_loc_grad[0][10],hidden_loc_grad[0][11],hidden_loc_grad[0][12],hidden_loc_grad[0][13],hidden_loc_grad[0][14],hidden_loc_grad[0][15],hidden_loc_grad[0][16]])#this is a 1X16 vector which removes the bias neuron
        #print(hidden_loc_grad1)
        # now to update the weights after every Iterations
        learn = 0.001
        alpha = 0.6
        #first we update the weights matrix 'a'
        for r in range(7):
            for c in range(16):
                a[r][c] = a[r][c] + ((a_list[count][r][c] - a_list[count-1][r][c])*alpha) + learn*hidden_loc_grad1[c]*temp_inp[r]# includes the momentum term

        # now we update the weights matrix 'b'
        for r in range(17):
            for c in range(1):
                b[r][c] = b[r][c] + ((b_list[count][r][c] - b_list[count-1][r][c])*alpha) + learn*output_loc_grad*hidden_activation[r]#includes the momentum term
        a_list.append(a)
        b_list.append(b)
    # compute variance and plot Error
    #print("Matrix A: ",a)
    #print("Matrix B:", b)
    if epoch % 50 == 0:
        print(epoch)
        mse = 0
        for j in range(99005,100000):
            temp_inp = np.array([1,v[j],v[j-1],v[j-2],v[j-3],v[j-4],v[j-5]])#1X7 vector
            #this is for the forward propagation
            hidden_output = np.dot(temp_inp,a)#1X16 vector
            hidden_activation = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype = 'float')
            for k in range(1,17):
                hidden_activation[k] = 1/(1 + np.exp(-hidden_output[k-1]))#1X17 vector(first element is the bias), these serve as inputs for the output layer
            output = np.dot(hidden_activation,b)#this is a 1X1 matrix
            output = output[0]
            des = v[j] + (0.5*v[j-1]*v[j-2])
            mse = mse + (output - des)*(output - des)
        error.append(mse)
# plotting
x_list = []
for i in range(1, 1001):
    if i%50 == 0:
       x_list.append(i)
plt.scatter(x_list,error)
plt.plot(x_list,error)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Error Trajectory-Cross Validation")
plt.show()
'''
