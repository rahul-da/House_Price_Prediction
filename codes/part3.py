#PART 3
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def sum_of_squares(arr):
	return(sqrt(np.matmul(arr,np.transpose(arr))))
	
def hypothesis(X, theta):
	return(np.matmul(X,theta))


def gradient_descent(theta, X, y, m, learning_rate):
	xTrans = X.transpose()
	hypothesis = np.dot(X, theta)
	loss = hypothesis - y
	cost = np.sum(loss ** 2) / (2 * m)
	#print("Iteration %d | Cost: %f" % (i, cost))
	gradient = np.dot(xTrans, loss) / m
	theta = theta - learning_rate * gradient
	return(theta)
	


if __name__ == '__main__':
	l = 'kc_house_data.csv.xlsx'
	
	df = pd.read_excel('kc_house_data.csv.xlsx')

	df.train = df.iloc[:17290, :]			#80% FOR TRAINING
	df.test = df.iloc[17290:, :]			#20% FOR TESTING


	print(df)					#Entire DATASET
	print(df.train)				#Training DATASET
	print(df.test)				#Test DATASET
	
	
	#LINEAR PARAMETERS
	theta = np.zeros(5)			#Initialization of Model Parameters
	m = 17290				#Number of Training Examples
	lr = [0.01, 0.05, 0.1, 0.2, 0.5]		#Learning Rate for Gradient Descent
	lrRMSE = []
	theta = np.zeros(13)			#Initialization of Model Parameters
	m = 17290				#Number of Training Examples
	lr = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]		#Learning Rate for Gradient Descent
	#lr = [0.05]
	lrRMSE = []
	for learning_rate in lr:
		theta = np.zeros(5)
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
		#print(X)
		#print(y)
	
		#print(X[1])
		#print(hypothesis(X[1],theta))
	
		iteration = []
		RMSE = []
		RMSET =[]
		for i in range(500):
			#print("Theta",theta)
			theta = gradient_descent(theta, X, y, m, learning_rate)
			s = 0
			for j in range(m):
				s += (norm_min+(norm-norm_min)*hypothesis(X[j],theta)-norm_min-(norm-norm_min)*y[j])**2
			#print("Training mean square Error at ",i+1,"iteration:",s/(2*m))
			iteration.append(i+1)
			RMSE.append(sqrt(s/m))
			s = 0
			for j in range(4323):	
				#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
				s += (norm_min+(norm-norm_min)*hypothesis(X_test[j],theta)-y_test[j])**2
			RMSET.append(sqrt(s/4323))
		#print(theta)
	
		s = 0
		for i in range(m):	
			#print(norm_min+(norm-norm_min)*hypothesis(X[i],theta),norm_min+(norm-norm_min)*y[i])
			s += (norm_min+(norm-norm_min)*hypothesis(X[i],theta)-norm_min+(norm-norm_min)*y[i])**2
		ss = sqrt(s/m)
		#print("Training RMSE : ",ss)
		#(norm_min+(norm-norm_min)*hypothesis(X[1],theta),norm_min+(norm-norm_min)*y[1])
		#print(norm_min+(norm-norm_min)*hypothesis(X[2],theta),norm_min+(norm-norm_min)*y[2])
		s=0
		for i in range(4323):	
			#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
			s += (norm_min+(norm-norm_min)*hypothesis(X_test[i],theta)-y_test[i])**2
		sss = sqrt(s/4323)
		print("Testing RMSE : ",sss)
		lrRMSE.append(sss)
		#Training RMSE :  1011686.6457380073
		#Testing RMSE :  344826.9810565903

	"""
	plt.plot(iteration , RMSE , label="TRAIN" )
	plt.plot(iteration , RMSET,label = "TEST")
	plt.legend()
	plt.show()
	"""
	print("A0 = ",theta[0],"A1 = ",theta[1],"A2 = ",theta[2],"A3 = ",theta[3],"A4 = ",theta[4])
	plt.plot(lr,lrRMSE,label="LINEAR")
	#plt.show()
	
	#QUADRATIC PARAMETERS
	theta = np.zeros(9)			#Initialization of Model Parameters
	m = 17290				#Number of Training Examples
	lr = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]		#Learning Rate for Gradient Descent
	lrRMSE = []
	#lr = [0.05]
	for learning_rate in lr:
		theta = np.zeros(9)
		bias = np.array([np.ones(17290)])		#1.0 for Bias Coeficients
		bias_test = np.array([np.ones(4323)])
		mx = np.ones(5)
		mn = np.ones(5)
		dd = df.train.as_matrix()
		dd_test = df.test.as_matrix()
		data = np.concatenate((bias.T, dd), axis=1)
		testdata = np.concatenate((bias_test.T, dd_test), axis=1)
		X = data[:,[0,1,2,3,4,1,2,3,4]]
		y = data[:,5]
		X_test = testdata[:,[0,1,2,3,4,1,2,3,4]]
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
		X[:,5] = X[:,1] * X[:,1]
		X[:,6] = X[:,2] * X[:,2]
		X[:,7] = X[:,3] * X[:,3]
		X[:,8] = X[:,4] * X[:,4]
	
		X_test[:,5] = X_test[:,1] * X_test[:,1]
		X_test[:,6] = X_test[:,2] * X_test[:,2]
		X_test[:,7] = X_test[:,3] * X_test[:,3]
		X_test[:,8] = X_test[:,4] * X_test[:,4]
		#print(X)
		#print(y)
	
		#print(X[1])
		#print(hypothesis(X[1],theta))
	
		iteration = []
		RMSE = []
		RMSET =[]
		for i in range(500):
			#print("Theta",theta)
			theta = gradient_descent(theta, X, y, m, learning_rate)
			s = 0
			for j in range(m):
				s += (norm_min+(norm-norm_min)*hypothesis(X[j],theta)-norm_min-(norm-norm_min)*y[j])**2
			#print("Training mean square Error at ",i+1,"iteration:",s/(2*m))
			iteration.append(i+1)
			RMSE.append(sqrt(s/m))
			s = 0
			for j in range(4323):	
				#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
				s += (norm_min+(norm-norm_min)*hypothesis(X_test[j],theta)-y_test[j])**2
			RMSET.append(sqrt(s/4323))
	
		#theta = gradientDescent(X, y, theta, learning_rate, m, 500)
		#print(theta)
	
		s = 0
		for i in range(m):	
			#print(norm_min+(norm-norm_min)*hypothesis(X[i],theta),norm_min+(norm-norm_min)*y[i])
			s += (norm_min+(norm-norm_min)*hypothesis(X[i],theta)-norm_min+(norm-norm_min)*y[i])**2
		ss = sqrt(s/m)
		#print("Training RMSE : ",ss)
		#(norm_min+(norm-norm_min)*hypothesis(X[1],theta),norm_min+(norm-norm_min)*y[1])
		#print(norm_min+(norm-norm_min)*hypothesis(X[2],theta),norm_min+(norm-norm_min)*y[2])
		s=0
		for i in range(4323):	
			#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
			s += (norm_min+(norm-norm_min)*hypothesis(X_test[i],theta)-y_test[i])**2
		sss = sqrt(s/4323)
		print("Testing RMSE : ",sss)
		lrRMSE.append(sss)
		#Training RMSE :  1011686.6457380073
		#Testing RMSE :  344826.9810565903

	"""
	plt.plot(iteration , RMSE , label="TRAIN" )
	plt.plot(iteration , RMSET,label = "TEST")
	plt.legend()
	plt.show()
	"""
	print("A0 = ",theta[0],"A1 = ",theta[1],"A2 = ",theta[2],"A3 = ",theta[3],"A4 = ",theta[4],"A5 = ",theta[5],"A6 = ",theta[6],"A7 = ",theta[7],"A8 = ",theta[8])
	plt.plot(lr,lrRMSE,label="QUADRATIC")
	#plt.show()	
	
	#CUBIC PARAMETERS 
	theta = np.zeros(13)			#Initialization of Model Parameters
	m = 17290				#Number of Training Examples
	lr = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]		#Learning Rate for Gradient Descent
	lrRMSE = []
	#lr = [0.05]
	
	theta = np.zeros(13)			#Initialization of Model Parameters
	m = 17290				#Number of Training Examples
	lrRMSE = []
	for learning_rate in lr:
		theta = np.zeros(13)
		bias = np.array([np.ones(17290)])		#1.0 for Bias Coeficients
		bias_test = np.array([np.ones(4323)])
		mx = np.ones(5)
		mn = np.ones(5)
		dd = df.train.as_matrix()
		dd_test = df.test.as_matrix()
		data = np.concatenate((bias.T, dd), axis=1)
		testdata = np.concatenate((bias_test.T, dd_test), axis=1)
		X = data[:,[0,1,2,3,4,1,2,3,4,1,2,3,4]]
		y = data[:,5]
		X_test = testdata[:,[0,1,2,3,4,1,2,3,4,1,2,3,4]]
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
		X[:,5] = X[:,1] * X[:,1]
		X[:,6] = X[:,2] * X[:,2]
		X[:,7] = X[:,3] * X[:,3]
		X[:,8] = X[:,4] * X[:,4]
		X[:,9] = X[:,1] * X[:,5]
		X[:,10] = X[:,2] * X[:,6]
		X[:,11] = X[:,3] * X[:,7]
		X[:,12] = X[:,4] * X[:,8]
	
		X_test[:,5] = X_test[:,1] * X_test[:,1]
		X_test[:,6] = X_test[:,2] * X_test[:,2]
		X_test[:,7] = X_test[:,3] * X_test[:,3]
		X_test[:,8] = X_test[:,4] * X_test[:,4]
		X_test[:,9] = X_test[:,1] * X_test[:,5]
		X_test[:,10] = X_test[:,2] * X_test[:,6]
		X_test[:,11] = X_test[:,3] * X_test[:,7]
		X_test[:,12] = X_test[:,4] * X_test[:,8]
		#print(X)
		#print(y)
	
		#print(X[1])
		#print(hypothesis(X[1],theta))
	
		iteration = []
		RMSE = []
		RMSET =[]
		for i in range(500):
			#print("Theta",theta)
			theta = gradient_descent(theta, X, y, m, learning_rate)
			s = 0
			for j in range(m):
				s += (norm_min+(norm-norm_min)*hypothesis(X[j],theta)-norm_min-(norm-norm_min)*y[j])**2
			#print("Training mean square Error at ",i+1,"iteration:",s/(2*m))
			iteration.append(i+1)
			RMSE.append(sqrt(s/m))
			s = 0
			for j in range(4323):	
				#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
				s += (norm_min+(norm-norm_min)*hypothesis(X_test[j],theta)-y_test[j])**2
			RMSET.append(sqrt(s/4323))
	
		#theta = gradientDescent(X, y, theta, learning_rate, m, 500)
		#print(theta)
	
		s = 0
		for i in range(m):	
			#print(norm_min+(norm-norm_min)*hypothesis(X[i],theta),norm_min+(norm-norm_min)*y[i])
			s += (norm_min+(norm-norm_min)*hypothesis(X[i],theta)-norm_min+(norm-norm_min)*y[i])**2
		ss = sqrt(s/m)
		#print("Training RMSE : ",ss)
		#(norm_min+(norm-norm_min)*hypothesis(X[1],theta),norm_min+(norm-norm_min)*y[1])
		#print(norm_min+(norm-norm_min)*hypothesis(X[2],theta),norm_min+(norm-norm_min)*y[2])
		s=0
		for i in range(4323):	
			#print(norm_min+(norm-norm_min)*hypothesis(X_test[i],theta),y_test[i])
			s += (norm_min+(norm-norm_min)*hypothesis(X_test[i],theta)-y_test[i])**2
		sss = sqrt(s/4323)
		print("Testing RMSE : ",sss)
		lrRMSE.append(sss)
		#Training RMSE :  1011686.6457380073
		#Testing RMSE :  344826.9810565903

	"""
	plt.plot(iteration , RMSE , label="TRAIN" )
	plt.plot(iteration , RMSET,label = "TEST")
	plt.legend()
	plt.show()
	"""
	print("A0 = ",theta[0],"A1 = ",theta[1],"A2 = ",theta[2],"A3 = ",theta[3],"A4 = ",theta[4],"A5 = ",theta[5],"A6 = ",theta[6],"A7 = ",theta[7],"A8 = ",theta[8],"A9 = ",theta[9],"A10 = ",theta[10],"A11 = ",theta[11],"A12 = ",theta[12])
	plt.plot(lr,lrRMSE,label="CUBIC")
	plt.legend()
	plt.show()
	
