#PART 2
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def sum_of_squares(arr):
	return(sqrt(np.matmul(arr,np.transpose(arr))))
	
def hypothesis(X, theta):
	return(np.matmul(X,theta))


def gradient_descent(theta, X, y, m, learning_rate):
	temp_theta = np.zeros(theta.size)
	for i in range(theta.size):
		temp_theta[i]  = theta[i]
	#Grad Computations
	for i in range(theta.size):
		s = 0
		for j in range(m):
			s += (hypothesis(X[j],theta) - y[j])*X[j][i] 
		temp_theta[i] -= (learning_rate *  s)  / m	
		#print(s/m)

	print(theta - temp_theta)
	#Simultaneous Updates
	return(temp_theta)


def IRLS(theta, X, y, m):
	H = np.matmul(np.transpose(X),X)
	Predict = np.dot(X, theta)
	loss = Predict - y
	cost = np.sum(loss**2) /(2*m)
	gradient = np.dot(X.transpose(), loss)
	theta = theta - np.matmul(np.linalg.inv(H) , gradient)
	return(theta)
	
	
def regularization(theta, X, y, m, learning_rate,regularisation_parameter):
	temp_theta = np.zeros(theta.size)
	for i in range(theta.size):
		temp_theta[i]  = theta[i]
	#Grad Computations
	for i in range(theta.size):
		s = 0
		for j in range(m):
			s += (hypothesis(X[j],theta) - y[j])*X[j][i] 
		if i==0:
			temp_theta[i] -= (learning_rate *  s)  / m	
		else:
			temp_theta[i] = temp_theta[i]*(1-learning_rate*regularisation_parameter/m) -(learning_rate *  s)  / m	
		#print(s/m)

	print(theta - temp_theta)
	#Simultaneous Updates
	return(temp_theta)	
	
	


if __name__ == '__main__':
	l = 'kc_house_data.csv.xlsx'
	
	df = pd.read_excel('kc_house_data.csv.xlsx')

	df.train = df.iloc[:17290, :]			#80% FOR TRAINING
	df.test = df.iloc[17290:, :]			#20% FOR TESTING


	print(df)					#Entire DATASET
	print(df.train)				#Training DATASET
	print(df.test)				#Test DATASET
	
	theta = np.zeros(5)			#Initialization of Model Parameters
	m = 17290				#Number of Training Examples
	learning_rate = 0.05		#Learning Rate for Gradient Descent
	
	bias = np.array([np.ones(17290)])		#1.0 for Bias Coeficients
	bias_test = np.array([np.ones(4323)])
	mx = np.ones(5)
	mn = np.ones(5)
	dd = df.train.as_matrix()
	dd_test = df.test.as_matrix()
	data = np.concatenate((bias.T, dd), axis=1)
	testdata = np.concatenate((bias_test.T, dd_test), axis=1)
	X = data[:,[0,1,2,3,4]]
	y = data[:,5]
	X_test = testdata[:,[0,1,2,3,4]]
	y_test = testdata[:,5] 
	norm = max(y)
	norm_min = min(y)
	y = (y - norm_min)/(norm-norm_min)
	print(norm)
	for i in range(5):
		mx[i] = max(X[:,i])
		mn[i] = min(X[:,i])
	for i in range(5):
		if i==0:
			continue;
		for j in range(m):
			X[j,i] = (X[j,i]-mn[i])/(mx[i]-mn[i])
		for j in range(4323):
			X_test[j,i] = (X_test[j,i]-mn[i])/(mx[i]-mn[i])
	print(X)
	print(y)
	
	print(X[1])
	print(hypothesis(X[1],theta))
	
	iteration = []
	RMSE = []
	RMSET = []
	for i in range(50):
		#theta0 = gradient_descent(theta, X, y, m, 0.05)
		print("Theta at iteration ",i,theta)
		theta = IRLS(theta, X, y, m)
		s = 0
		for j in range(m):
			s += (norm_min+(norm-norm_min)*hypothesis(X[j],theta)-norm_min-(norm-norm_min)*y[j])**2
		print("Training mean square Error at ",i+1,"iteration:",s/(2*m))
		iteration.append(i+1)
		RMSE.append(sqrt(s/m))
		s = 0
		for j in range(4323):	
			#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
			s += (norm_min+(norm-norm_min)*hypothesis(X_test[j],theta)-y_test[j])**2
		RMSET.append(sqrt(s/4323))
	
	s = 0
	for i in range(m):	
		#print(norm_min+(norm-norm_min)*hypothesis(X[i],theta),norm_min+(norm-norm_min)*y[i])
		s += (norm_min+(norm-norm_min)*hypothesis(X[i],theta)-norm_min-(norm-norm_min)*y[i])**2
	ss = sqrt(s/m)
	print("Training RMSE : ",ss)
	#(norm_min+(norm-norm_min)*hypothesis(X[1],theta),norm_min+(norm-norm_min)*y[1])
	#print(norm_min+(norm-norm_min)*hypothesis(X[2],theta),norm_min+(norm-norm_min)*y[2])
	s=0
	for i in range(4323):	
		#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
		s += (norm_min+(norm-norm_min)*hypothesis(X_test[i],theta)-y_test[i])**2
	sss = sqrt(s/4323)
	print("Testing RMSE : ",sss)
	
	plt.plot(iteration , RMSE , label="TRAIN" )
	plt.plot(iteration , RMSET,label = "TEST")
	plt.legend()
	plt.show()
	
	#Parameters [-0.01238847  0.08170224  0.00660119  0.07561166  0.24383793]
	
	
	#Training RMSE :  311072.36852142174
	#Testing RMSE :  314733.65434302576

	
	
	
