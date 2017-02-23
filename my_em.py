#coding:utf-8
import numpy as np
import math 
import matplotlib.pyplot as plt
import pylab as pl



def nDimGauss(X, mu, variance):
	dim = np.shape(X)[0]
	#print dim
	pi = math.pi
	exp = np.exp
	
	X = np.mat(X)
	mu = np.mat(mu)
	tmp = X-mu
	
	variance = np.mat(variance)
	#print variance
	deter = np.linalg.det(variance)
	#deter = 1
	tmp0 = np.linalg.pinv(variance)
	tmp1 = tmp*tmp0*tmp.transpose()
	#print tmp0,tmp1,deter,dim
	prob = 1/(((2*pi)**(dim/2))*(deter**0.5))*exp(-0.5*tmp1)
	#print prob
	return prob

def init_para(c,dim,mu,variance):
	c = c
	ck = np.array(np.ones((c,1)))*(1/float(c)) 
	
	list_cov = []
	list_mu = []
	#mu_k = np.array((c, dim))

	for i in range(c):
		mu = np.random.rand(dim)
		mu = mu.reshape(1,dim)
		list_mu.append(mu)
	for i in range(c):
		list_cov.append(variance)
	#print variance.shape()
	# print (ck,list_mu,list_cov)
	return ck,list_mu,list_cov
	
def calFuction(Xn ,Ck, list_mu, list_cov):
	Xn = np.mat(Xn)
	num, dim = np.shape(Xn)
	#print num, dim
	k = len(list_mu)

	energy = 0.0

	for i in range(num):
		tmp = 0
		for j in range(k):
			#print nDimGauss(Xn[i,:], list_mu[j], list_cov[j])
			tmp += Ck[j] *  nDimGauss(Xn[i,:], list_mu[j], list_cov[j])
			#print tmp
		energy += np.log(tmp)
	return float(energy)	

def update(X, c_k, list_mu, list_cov):
	num, dim = np.shape(X)
	c,notused = np.shape(c_k)
	k = len(list_mu)
	Nk = np.array(np.zeros((k, 1)))
	r_zn_k = np.array(np.zeros((num,k)))
	list_cov_new = []
	list_mu_new = []
	c_k_new = np.array(np.zeros((c,1)))
	for n_iter in range(num):
		total = 0
		for k_iter in range(k):
			r_zn_k[n_iter, k_iter] = c_k[k_iter] * nDimGauss(X[n_iter], list_mu[k_iter], list_cov[k_iter])
			total += r_zn_k[n_iter, k_iter] 
		for k_iter in range(k):
			r_zn_k[n_iter, k_iter]  = r_zn_k[n_iter, k_iter]  / total	
	#print 'r_zn_k',r_zn_k,'\n'

	for k_iter in range(k):
		tmp_mu = np.array(np.zeros((1, dim)))
		#
		for n_iter in range(num):
			Nk[k_iter,0] += r_zn_k[n_iter, k_iter]
			tmp_mu +=  r_zn_k[n_iter, k_iter] * X[n_iter]
		list_mu_new.append(tmp_mu / Nk[k_iter, 0])
	#print 'list_mu_new',list_mu_new,'\n'

	for  k_iter in range(k):
		tmp_var = np.array(np.zeros((dim, dim)))
		for n_iter in range(num):
			tmp_var +=  r_zn_k[n_iter, k_iter] *((X[n_iter] - list_mu_new[k_iter]).transpose() * (X[n_iter] - list_mu_new[k_iter]))
		list_cov_new.append(tmp_var / Nk[k_iter, 0]) 
	#print 'list_cov_new',list_cov_new,'\n'


	for k_iter in range(k):
		c_k_new[k_iter] = Nk[k_iter] / num
	#print 'c_k_new',c_k_new,'\n'

	return r_zn_k, c_k_new, list_mu_new, list_cov_new
	


data1 = dict()
data2 = dict()

to_plot = []
to_plot2 = []

with open('./GMM-EM/train.txt') as fin:
	X1 = [];
	X2 = [];
	for line in fin:
		tmp = line.strip().split(" ")

		if tmp[-1]=='1' :
			X1.append([tmp[0],tmp[1]])
		if tmp[-1]=='2' :
			X2.append([tmp[0],tmp[1]])
	X1 = np.array(X1)
	X1 = X1.astype('float64') 
	X2 = np.array(X2)
	X2 = X2.astype('float64')
	#print X1[1][1];
	#mu = calMu(X1)
	mu1 = X1.mean(axis=0)
	#print mu1

	pl.plot(X1[:,0], X1[:,1], 'ro')
	pl.xlabel('x1')
	pl.ylabel('x2')
	pl.xlim(-4, 4)
	pl.show()

	pl.plot(X2[:,0], X2[:,1], 'ro')
	pl.xlabel('x1')
	pl.ylabel('x2')
	pl.xlim(-4, 4)
	pl.show()

	x1 = X1.transpose()
	var1 = np.cov(x1)

	mu2 = X2.mean(axis=0)
	x2 = X2.transpose()
	var2 = np.cov(x2)
	#print X1.shape
	# x1 = np.array([X1[0][0], X1[0][1]])
	# print var1
	# print np.linalg.det(var1)
	# print var1.shape
	
	# print np.linalg.pinv(var1) ;
	# #print np.linalg.pinv(var1)*var1 ;
	# print nDimGauss(x1, mu1, var1)
	# print mu1.shape
	
	#print ck
	Ck_1, list_mu_1, list_cov_1 = init_para(4, 2,mu1,var1)
	print Ck_1, list_mu_1, list_cov_1
	tmp = calFuction(X1 ,Ck_1, list_mu_1, list_cov_1)
	print tmp
	r_zn_k, c_k_new, list_mu_new, list_cov_new = update(X1, Ck_1, list_mu_1, list_cov_1)

	itera = 40
	while itera > 0  :
		tmp = calFuction(X1 ,c_k_new, list_mu_new, list_cov_new)
		print tmp
		to_plot.append(tmp)
		r_zn_k, c_k_new, list_mu_new, list_cov_new = update(X1, c_k_new, list_mu_new, list_cov_new)

		itera -=1
	horizen = xrange(0,40)

	pl.plot(horizen, to_plot, 'ro')
	pl.title('Plot of log likelihood')# give plot a title
	pl.xlabel('iteration')# make axis labels
	pl.ylabel('log likelihood')
	pl.show()

	Ck_2, list_mu_2, list_cov_2 = init_para(4, 2,mu2,var2)
	print Ck_2, list_mu_2, list_cov_2
	tmp = calFuction(X2 ,Ck_2, list_mu_2, list_cov_2)
	print tmp
	r_zn_k_2, c_k_new_2, list_mu_new_2, list_cov_new_2 = update(X2, Ck_2, list_mu_2, list_cov_2)
	itera = 40
	while itera > 0 :
		r_zn_k_2, c_k_new_2, list_mu_new_2, list_cov_new_2 = update(X2, c_k_new_2, list_mu_new_2, list_cov_new_2)
		tmp = calFuction(X2 ,c_k_new_2, list_mu_new_2, list_cov_new_2)
		print tmp
		to_plot2.append(tmp)
		itera -=1

	pl.plot(horizen, to_plot2, 'ro')
	pl.title('Plot of log likelihood')# give plot a title
	pl.xlabel('iteration')# make axis labels
	pl.ylabel('log likelihood')
	pl.show()


with open('./GMM-EM/dev.txt') as fin1:
	dev = []
	result1 = []
	result2 = []
	for line in fin1:
		tmp = line.strip().split(" ")
		dev.append([tmp[0],tmp[1]])
	dev = np.array(dev)
	dev = dev.astype('float64')
	total_num,dim = np.shape(dev)
	num_of_gauss = 4
	for i in range(total_num):
		total = 0
		for k in range(num_of_gauss):
			total +=c_k_new[k] * nDimGauss(dev[i], list_mu_new[k], list_cov_new[k])
		result1.append(total)
	for i in range(total_num):
		total = 0
		for k in range(num_of_gauss):
			total +=c_k_new_2[k] * nDimGauss(dev[i], list_mu_new_2[k], list_cov_new_2[k])
		result2.append(total)


result = []
num = len(result1)
for i in range(num):
	if result1[i]>result2[i] :

		result.append(1)
	else :
		result.append(2)

file1 = open('./GMM-EM/dev.txt','r')
file2 = open('./result_dev.txt','w')

i = 0
for line in file1:
	tmp = line.strip()
	file2.write(tmp + "\t" + str(result[i]) + "\n")
	i += 1

with open('./GMM-EM/test.txt') as fin1:
	dev = []
	result1 = []
	result2 = []
	for line in fin1:
		tmp = line.strip().split(" ")
		dev.append([tmp[0],tmp[1]])
	dev = np.array(dev)
	dev = dev.astype('float64')
	total_num,dim = np.shape(dev)
	num_of_gauss = 4
	for i in range(total_num):
		total = 0
		for k in range(num_of_gauss):
			total +=c_k_new[k] * nDimGauss(dev[i], list_mu_new[k], list_cov_new[k])
		result1.append(total)
	for i in range(total_num):
		total = 0
		for k in range(num_of_gauss):
			total +=c_k_new_2[k] * nDimGauss(dev[i], list_mu_new_2[k], list_cov_new_2[k])
		result2.append(total)


result = []
num = len(result1)
for i in range(num):
	if result1[i]>result2[i] :

		result.append(1)
	else :
		result.append(2)

file1 = open('./GMM-EM/test.txt','r')
file2 = open('./result_test.txt','w')
i = 0
for line in file1:
	tmp = line.strip()
	file2.write(tmp + "\t" + str(result[i]) + "\n")
	i += 1
#np.savetxt('./result_new.txt', result )




	
