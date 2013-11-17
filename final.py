import os
from PIL import Image
import numpy as np
import math
import random
import string

random.seed(0)
#Constants
###########
#Directory of training data set folder
train_dir='/home/vr/Desktop/PR PRo/HR/New3_3/'
test_dir='/home/vr/Desktop/PR PRo/HR/Test3_3/'



#Functions
###########
# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2
def get_trainigset(path):
	lis=[]
	for dataset in os.listdir(path):
		img = np.asarray(Image.open(path+dataset).convert('L'))
		if dataset[0]=='N':
			target=[1,0,0,0]
		elif dataset[0]=='I':
			target=[0,1,0,0]
		elif dataset[0]=='T':
			target=[0,0,1,0]
		elif dataset[0]=='H':
			target=[0,0,0,1]

		img = 1 * (img < 127)
		h,w = img.shape
		h1=h/3
		h2=h1*2
		w1=w/3
		w2=w1*2
		total_pix=img.sum()

		raws=[	
				[img[0:h1,0:w1].sum(),#F1 for block one
				img[0:h1,w1:w2].sum(),#
				img[0:h1,w2:w].sum(),#
				img[h1:h2,0:w1].sum(),#
				img[h1:h2,w1:w2].sum(),# f5 for total one
				img[h1:h2,w2:w].sum(),#
				img[h2:h,0:w1].sum(),#
				img[h2:h,w1:w2].sum(),#
				img[h2:h,w2:w].sum(),#F9
				],
				target		
			]
		lis.append(raws)
	maxs=[0]*9
	for rws in lis:
		for i in range(0,9):
			if maxs[i] < rws[0][i]:
				maxs[i] = rws[0][i]
	#print maxs
	for rws in lis:
		for i in range(0,9):
			rws[0][i]=float(rws[0][i])/float(maxs[i])
	return lis
def get_testingset(path):
	lis=[]
	for dataset in os.listdir(path):
		img = np.asarray(Image.open(path+dataset).convert('L'))
		if dataset[0]=='N':
			target=[1,0,0,0]
		elif dataset[0]=='I':
			target=[0,1,0,0]
		elif dataset[0]=='T':
			target=[0,0,1,0]
		elif dataset[0]=='H':
			target=[0,0,0,1]

		img = 1 * (img < 127)
		h,w = img.shape
		h1=h/3
		h2=h1*2
		w1=w/3
		w2=w1*2
		total_pix=img.sum()

		raws=[	
				[img[0:h1,0:w1].sum(),#F1 for block one
				img[0:h1,w1:w2].sum(),#
				img[0:h1,w2:w].sum(),#
				img[h1:h2,0:w1].sum(),#
				img[h1:h2,w1:w2].sum(),# f5 for total one
				img[h1:h2,w2:w].sum(),#
				img[h2:h,0:w1].sum(),#
				img[h2:h,w1:w2].sum(),#
				img[h2:h,w2:w].sum(),#F9
				],
				[dataset]		
			]
		lis.append(raws)
	maxs=[0]*9
	for rws in lis:
		for i in range(0,9):
			if maxs[i] < rws[0][i]:
				maxs[i] = rws[0][i]
	#print maxs
	for rws in lis:
		for i in range(0,9):
			rws[0][i]=float(rws[0][i])/float(maxs[i])
	return lis
def show_chars(liss):
	hash_tb=['N','I','T','H']
	for i in liss:
		j=i[1].index(max(i[1]))
		print "File %s is letter %s"%(i[0][0],hash_tb[j])

#class ffor neural network
class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
    	res=[]
        for p in patterns:
            print(p[0], '->', self.update(p[0]))
            res.append([p[1],self.update(p[0])])
        return res

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)
def demo():
	train_data=get_trainigset(train_dir)
	test_data=get_testingset(test_dir)
	n = NN(9, 10, 4)
	n.train(train_data,2000,.5,.01)
	ress=n.test(test_data)
	print "Final Results"
	print "*"*20
	#print ress
	show_chars(ress)
	#print train
	#print test

if __name__ == '__main__':
    demo()