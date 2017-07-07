import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from matplotlib.colors import LogNorm

class EM:

    def __init__(self, samples, cluster_no,threshold):
        self.K = cluster_no  # number of gaussian distribution
        self.th = threshold
        self.X = samples
        
    def gaussian_distribution(self,miu, sigma):

        m = (np.linalg.det(sigma)**(-.5)) * ((2 * np.pi)**(-self.K/2.0))
        
        n = np.dot(np.linalg.inv(sigma) , (self.X - miu).T) #(2 x 2)*(2 x 272) = (2 x 272)
        p = np.einsum('ij, ij -> i', self.X-miu, n.T) #element wise multiplication(272 x 2)*(272,2) = (272 x1)
        P = m * np.exp(-.5 * p) #(272 x 1)
        return P

        
    def execute_EM(self, max_iterations):

        row ,dimension = self.X.shape # (row X dimension)= (272,2) for this sample data

        #means for each gaussian
        samples_index = np.random.choice(row, self.K, False) 
        miu = self.X[samples_index, :]
       
        #covariance matrix for each gaussian ([dimension x dimension] for each of K)
        sigma = [np.eye(dimension)]* self.K
       
        #probability for each gasussian (1 x 3)
        theta = [1.0/self.K] * self.K   
        
        #probability of samples for each of gaussian (total_sample x total_gaussian)
        pi_i_k = np.zeros( (row,self.K) ) 

        log_likelihoods =[]
       
      
        
        while len(log_likelihoods) < max_iterations :
            
            #E step<------------------->
            for k in range(self.K):
                pi_i_k[:,k] = theta[k] * self.gaussian_distribution(miu[k], sigma[k])

            log_likelihood = np.sum( np.log( np.sum( pi_i_k, axis = 1)))
       
            log_likelihoods.append(log_likelihood)

            pi_i_k = (pi_i_k.T / np.sum(pi_i_k, axis = 1)).T

            sum_pi_i_k = np.sum(pi_i_k, axis =0)

            #M step<------------------->
            for k in range(self.K):
                miu[k] = (1./sum_pi_i_k[k])* np.sum(pi_i_k[:,k]*X.T, axis = 1).T
               
                x_minus_miu = np.matrix(X - miu[k])   
                
                sigma[k] = np.array((1.0 / sum_pi_i_k[k]) * np.dot(np.multiply(x_minus_miu.T,  pi_i_k[:, k]), x_minus_miu))
 
                theta[k] = 1. / row * sum_pi_i_k[k]
              
             
            #Checking for convergence<--->
            if len(log_likelihoods)<2:
                continue
            if np.abs(log_likelihood - log_likelihoods[-2])< self.th:
                print("Break")
                break
            
        return miu,sigma


        
if __name__=="__main__":
    
    X=np.genfromtxt('loc.txt')
   
    plt.plot(X[:,0],X[:,1],'.')
    
   
    k = 2
    th = 0.0001
    iterations = 10000
    em = EM(X,k,th)
    miu, sigma = em.execute_EM(iterations)
   
    print(miu)
    print(sigma)
    for i in range(len(miu)):
        plt.scatter(miu[i][0], miu[i][1],color='red')
    
    plt.show()
    
  
 
   
