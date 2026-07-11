import numpy as np
class NN:
    def __init__(self,inp,hid,out,l):
        self.w_lst = []
        self.b_lst = []
        self.v_lst = []
        self.bv_lst = []
        self.wf_lst = []
        self.bf_lst = []
        self.h_lst = []
        self.z_lst = []
        self.de_lst = []
        for i in range(0,l+1):
            if i ==0:
                w = np.random.randn(inp,hid)*np.sqrt(2/inp)
                b = np.zeros((1,hid))
                v = np.zeros_like(w)
                bv = np.zeros_like(b)
                self.w_lst.append(w)
                self.b_lst.append(b)
                self.v_lst.append(v)
                self.bv_lst.append(bv)
                self.wf_lst.append(v)
                self.bf_lst.append(bv)
            elif i == l:
                w = np.random.randn(hid,out)*np.sqrt(2/hid)
                b = np.zeros((1,out))
                v = np.zeros_like(w)
                bv = np.zeros_like(b)
                self.w_lst.append(w)
                self.b_lst.append(b)
                self.v_lst.append(v)
                self.bv_lst.append(bv)
                self.wf_lst.append(v)
                self.bf_lst.append(bv)
            else:
                w = np.random.randn(hid,hid)*np.sqrt(2/hid)
                b = np.zeros((1,hid))
                v = np.zeros_like(w)
                bv = np.zeros_like(b)
                self.w_lst.append(w)
                self.b_lst.append(b)
                self.v_lst.append(v)
                self.bv_lst.append(bv)
                self.wf_lst.append(v)
                self.bf_lst.append(bv)
    def relu(self,x):
        return np.where(x>0,x,0.01*x)
    def relu_derivative(self,x):
        return np.where(x>0,1,0.01)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sigmoid_derivative(self,x):
        return x*(1-x)
    def train(self,x,y,lr = 0.01,g=0.9,epochs = 10000):
        for i in range(epochs):
            for j in range(len(self.w_lst)):
                self.wf_lst[j] = self.w_lst[j] - self.v_lst[j]*g
                self.bf_lst[j] = self.b_lst[j] - self.bv_lst[j]*g
            h = x@self.wf_lst[0] + self.bf_lst[0]
            self.h_lst.append(h)
            z = self.relu(h)
            self.z_lst.append(z)
            for k in range(1,len(self.w_lst)):
                h = z@self.wf_lst[k] + self.bf_lst[k]
                self.h_lst.append(h)
                if k !=len(self.w_lst)-1:
                    z = self.relu(h)
                    self.z_lst.append(z)
                else:
                    y_pred = self.sigmoid(h)
            e = y_pred - y
            de = e*self.sigmoid_derivative(y_pred)
            self.de_lst.append(de)
            for l in range(1,len(self.w_lst)):
                e = de@self.wf_lst[-l].T
                de = e*self.relu_derivative(self.h_lst[-l-1])
                self.de_lst.append(de)
            self.de_lst = self.de_lst[::-1]
            grad = x.T@self.de_lst[0]
            self.v_lst[0] = grad*lr + self.v_lst[0]*g
            self.w_lst[0]-=self.v_lst[0]
            grad = np.sum(self.de_lst[0] , axis = 0, keepdims= True)
            self.bv_lst[0] = grad*lr + self.bv_lst[0]*g
            self.b_lst[0]-=self.bv_lst[0]
            for m in range(1,len(self.w_lst)):
                grad = self.z_lst[m-1].T@self.de_lst[m]
                self.v_lst[m] = grad*lr + self.v_lst[m]*g
                self.w_lst[m]-=self.v_lst[m]
                grad = np.sum(self.de_lst[m],axis = 0,keepdims= True)
                self.bv_lst[m] = grad*lr + self.bv_lst[m]*g
                self.b_lst[m]-=self.bv_lst[m]
            if i %1000 ==0:
                l = np.mean(np.square(y_pred - y))
                print(f"the loss is {l}")
            self.de_lst=[]
            self.h_lst=[]
            self.z_lst=[]
x = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([[0,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[1,1]])
a = int(input("enter number of neurons in each hidden layer:"))
b = int(input("enter number of hidden layers:"))
model = NN(3,a,2,b)
model.train(x,y)
