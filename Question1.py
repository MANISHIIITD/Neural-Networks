
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']
    

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
      

        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        self.n_layers=n_layers
        self.layer_sizes=layer_sizes
        self.activation=activation
        self.learning_rate=learning_rate
        self.weight_init=weight_init
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        

        pass

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return  X * (X>=0)
        

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1*(X>=0)


    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1+np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        sgd=self.sigmoid(X)
        return  sgd*(1-sgd)
       

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones(X.shape)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1-np.tanh(X)*np.tanh(X)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        
        return np.exp(X)/(np.sum(np.exp(X),axis = 1, keepdims = True))

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros(shape)

    def random_init(self, shape):
        
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.rand(shape[0], shape[1])*0.01

    def normal_init(self, shape):
        
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        k=np.random.normal(size = shape, scale = 0.01)
        return k
        
    def one_hot_encoding(self,y):
        y1 = np.zeros((len(y),np.max(y)+1))
        for i in range(0,len(y)):
            y1[i,y[i]] = 1
        return y1
        
    
    def fit_batch_grad(self,X,y,x_test=None,y_test=None):
        
        
        m , n_0 = X.shape
        n_l = y.shape[1]

        weights,biases = self.initialize_func()
        self.weights = weights
        self.biases=biases

        train_loss_history = []
        
        test_loss_history = []
        

        

        for epoch in tqdm(range(self.num_epochs), desc = "Progress Total : ", position = 0, leave = True):


            n_batch = m//self.batch_size
            X_batch = [X[self.batch_size*i:self.batch_size*(i+1),:] for i in range(0,n_batch)]
            y_batch = [y[self.batch_size*i:self.batch_size*(i+1),:] for i in range(0,n_batch)]

            batch_loss_train = []
            batch_loss_test = []
            

            for currx, curry in tqdm(zip(X_batch,y_batch), desc = "Progress Epoch: " + str(epoch+1) + "/" + str(self.num_epochs), position = 0, leave = True, total = len(X_batch)):
                A, activation, preactivation = self.forward(currx,weights,biases)
                
                
                batch_loss_train.append(self.cross_entropy(A,curry))

                self.backward(currx,curry, activation,preactivation )

                if(x_test is not None):
                    proba = self.predict_proba(x_test)
                   
                    testloss = self.cross_entropy(proba, self.one_hot_encoding(y_test))
                    batch_loss_test.append(testloss)

            print("Validation loss = " ,np.array(batch_loss_test).mean())
            print("Training Loss = ", np.array(batch_loss_train).mean())
            


            train_loss_history.append( np.array(batch_loss_train).mean())

            test_loss_history.append( np.array(batch_loss_test).mean())

                
                
        
        self.train_loss_history = train_loss_history
        
        self.test_loss_history = test_loss_history
        
        
        self.weights = weights
        self.biases=biases
        


        return self
    def fit(self, X, y,x_test=None,y_test=None):

        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        
        y1 = self.one_hot_encoding(y)
        self.fit_batch_grad(X,y1,x_test,y_test)
        


        
        return self
        

    def forward(self,X,weights,biases):
        


        
    
        a=X
        activation={}
        preactivation={}
        for i in range(len(weights)-1):
            
                
              
            z=np.dot(a,weights[str(i+1)])+biases[str(i+1)]
            preactivation[str(i+1)]=z
            if(self.activation == "relu"):
                a = self.relu(z)

            elif (self.activation == "tanh"):
                a = self.tanh(z)

            elif (self.activation == "linear"):
                a = self.linear(z)

            elif (self.activation == "sigmoid"):
                a = self.sigmoid(z)
            
            
            activation[str(i+1)]=a
            
         
        Z_last = np.dot(a, weights[str(len(weights))]) + biases[ str(len(weights))]        
        A_last = self.softmax(Z_last) 
        
        
        activation[str(len(weights))] = A_last
        preactivation[ str(len(weights))] = Z_last
            
        
        return A_last,activation,preactivation
        

        
            
                
        
        
        
      
    def backward(self,X,y,activation,preactivation):
        
        derw={}
        derb={}
        Xlen=len(X)
        actlen = len(activation)
        activation["0"] = X
        A = activation[str(actlen)]
        dZ = A - y
        derw[str(actlen)] = np.dot(activation[str(actlen-1) ].T, dZ)/Xlen
        derb[str(actlen)] = np.sum(dZ, axis=0, keepdims=True) /Xlen
        

        devaprev = np.dot(dZ, self.weights[str(actlen)].T)

        
        
        #print(actlen)

        for i in range(actlen - 1, 0, -1):
            
            if(self.activation == "relu"):
                dt = self.relu_grad(preactivation[str(i)])

            elif (self.activation == "tanh"):
                dt = self.tanh_grad(preactivation[str(i)])

            elif (self.activation == "linear"):
                dt = self.linear_grad(preactivation[str(i)])

            elif (self.activation == "sigmoid"):
                dt = self.sigmoid_grad(preactivation[str(i)])
                
            #print(i)
            

            dZ =devaprev*dt
            dW = (1/Xlen) * np.dot(activation[str(i-1)].T, dZ)
            db = (1/Xlen) * np.sum(dZ,keepdims=True,axis=0)
            
            devaprev = np.dot(dZ,self.weights[str(i)].T)
                
                
            #print("i value")
            #print(i)
            #print(type(db))
            
        
            #print(type(dW))
            
            derw[str(int(i))]=dW
            #print(i)
            derb[str(int(i))]=db
            
    

        for i in range(0,actlen):
            self.weights[str(i+1)] = self.weights[str(i+1)] - self.learning_rate*derw[str(i+1)]
            self.biases[str(i+1)] = self.biases[str(i+1)] - self.learning_rate*derb[str(i+1)]
        return 
        
      
    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """
        prob,act,preact = self.forward(X,self.weights,self.biases)
        return prob
        
        
        

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """ 
     
        y_pred = np.argmax(self.predict_proba(X), axis = 1)
        return y_pred

        
    def initialize_func(self):
        
        weights={}
        biases={}
        layers = self.layer_sizes
        lcount=self.n_layers
        for i in range(0,lcount-1):
            if(self.weight_init == 'zero'):
                currentlayer = self.zero_init((layers[i],layers[i+1]))

            elif(self.weight_init == 'random'):
                currentlayer = self.random_init((layers[i],layers[i+1]))

            elif(self.weight_init == 'normal'):
                currentlayer = self.normal_init((layers[i],layers[i+1]))

            
            weights[str(i+1)]=currentlayer
            biases[str(i+1)]=np.zeros((1,layers[i+1]))
        self.weights=weights
        self.biases=biases

        

        return weights,biases
        

    def cross_entropy(self,a,y):
        
        
        
        plog = - np.log(a[np.arange(len(y)), y.argmax(axis=1)])
        
        cost = np.sum(plog)/len(y)
        return cost
    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """
        
        correct=0
        y_pred=self.predict(X)
        for i in range(0,len(y_pred)):
            if y[i]==y_pred[i]:
                correct=correct+1
        
            
        return correct/len(y)
    
    


    


# In[2]:


train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')


dataset = train_df.to_numpy()
testset = test_df.to_numpy()

from sklearn.preprocessing import StandardScaler
X_train = dataset[:, 1:]/255
X_test = testset[:, 1:]/255
standardscalar = StandardScaler()
X_train = standardscalar.fit_transform(X_train)
X_test = standardscalar.transform(X_test)

y_train = dataset[:, 0]
y_test = testset[:, 0]

    


# In[ ]:


batchsize=len(X_train)//25
nnr = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'relu', 0.1, 'normal',batchsize, 100)


nnr.fit(X_train, y_train, X_test, y_test)

accur=nnr.score(X_test,y_test)

print("Accuracy with Relu :")
print(accur)
import matplotlib.pyplot as plt

plt.plot([x for x in range(1,len(nnr.train_loss_history) + 1, 1)],nnr.train_loss_history, label = "Average Training  Loss " )
plt.plot([x for x in range(1,len(nnr.test_loss_history) + 1, 1)],nnr.test_loss_history, label = "Average Validation  Loss " )
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.legend()
plt.show()   


# In[ ]:



filename = 'relu_model.sav'
pickle.dump(nnr, open(filename, 'wb'))


# In[ ]:





batchsize=len(X_train)//25
nnt = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'tanh', 0.1, 'normal',batchsize, 100)


nnt.fit(X_train, y_train, X_test, y_test)

accut=nnt.score(X_test,y_test)

print("Accuracy with tanh :")
print(accut)



import matplotlib.pyplot as plt

plt.plot([x for x in range(1,len(nnt.train_loss_history) + 1, 1)],nnt.train_loss_history, label = "Average Training  Loss " )
plt.plot([x for x in range(1,len(nnt.test_loss_history) + 1, 1)],nnt.test_loss_history, label = "Average Validation  Loss " )
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.legend()
plt.show()       


# In[ ]:


filename = 'tanh_model.sav'
pickle.dump(nnt, open(filename, 'wb'))


# In[ ]:


batchsize=len(X_train)//25
nnl = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'linear', 0.1, 'normal',batchsize, 100)


nnl.fit(X_train, y_train, X_test, y_test)

accul=nnl.score(X_test,y_test)

print("Accuracy with linear :")
print(accul)



import matplotlib.pyplot as plt

plt.plot([x for x in range(1,len(nnl.train_loss_history) + 1, 1)],nnl.train_loss_history, label = "Average Training  Loss " )
plt.plot([x for x in range(1,len(nnl.test_loss_history) + 1, 1)],nnl.test_loss_history, label = "Average Validation  Loss " )
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.legend()
plt.show()       


# In[ ]:


filename1 = 'linear_model.sav'
pickle.dump(nnl, open(filename, 'wb'))


# In[ ]:


batchsize=len(X_train)//25
nns = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'sigmoid', 0.1, 'normal',batchsize, 100)


nns.fit(X_train, y_train, X_test, y_test)

accus=nns.score(X_test,y_test)

print("Accuracy with sigmoid :")
print(accus)



import matplotlib.pyplot as plt

plt.plot([x for x in range(1,len(nns.train_loss_history) + 1, 1)],nns.train_loss_history, label = "Average Training  Loss " )
plt.plot([x for x in range(1,len(nns.test_loss_history) + 1, 1)],nns.test_loss_history, label = "Average Validation  Loss " )
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.legend()
plt.show()       


# In[ ]:


filename1 = 'sigmoid_model.sav'
pickle.dump(nns, open(filename, 'wb'))


# In[ ]:


bestmod='tanh_model.sav'


# In[ ]:


best_model = pickle.load(open(bestmod, 'rb'))
res = best_model.score(X_test, y_test)
print("Accuracy")
res


# In[ ]:


Alast,activation,preactivation = best_model.forward(X_train,best_model.weights,best_model.biases)
print(activation.keys())
final_layer=activation['3']
len(final_layer)


# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(final_layer)
print(final_layer.shape,tsne[:,0].shape,tsne[:,1].shape)

plt.figure(figsize = (11,11))
plt.scatter(tsne[:,0],tsne[:,1])
plt.show()


# In[5]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


clf=MLPClassifier( solver='sgd', hidden_layer_sizes=(256,128,64), activation = 'relu',learning_rate='constant',learning_rate_init=0.1,max_iter=100)
clf.fit(X_train,y_train)
print("Relu Activation Score")
print(clf.score(X_test,y_test))


# In[6]:


clf=MLPClassifier( solver='sgd', hidden_layer_sizes=(256,128,64), activation = 'logistic',learning_rate='constant',learning_rate_init=0.1,max_iter=100)
clf.fit(X_train,y_train)
print("Sigmoid Activation Score")
print(clf.score(X_test,y_test))


# In[8]:


clf=MLPClassifier( solver='sgd', hidden_layer_sizes=(256,128,64), activation = 'identity',learning_rate='constant',learning_rate_init=0.1,max_iter=100)
clf.fit(X_train,y_train)
print("Linear Activation Score")
print(clf.score(X_test,y_test))


# In[9]:


clf=MLPClassifier( solver='sgd', hidden_layer_sizes=(256,128,64), activation = 'tanh',learning_rate='constant',learning_rate_init=0.1,max_iter=100)
clf.fit(X_train,y_train)
print(" Tanh Activation Score")
print(clf.score(X_test,y_test))


# In[ ]:




