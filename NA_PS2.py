import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

class QRsolver():
    ''' 
    This class solves the linear Ax=b
    It uses QR decomposition (Householder method).
    
    Input:
        np.array: A (m*n)
        np.array: B (m*1)
    '''
    
    
    def __init__(self, A, b):
        self.check_input(A, b)
        self.A = A
        self.b = b
        self.n_t=[]

    def check_input(self, A, b):
        assert(A.shape[0] >= A.shape[1])
        assert(b.shape == (A.shape[0], 1))
        #assert(A.linalg.det)
        
    def load_new(self, A, b):
        self.check_input(A, b)
        self.A = A
        self.b = b
        
        
    def QR_factorization(self):
        R = self.A.copy()
        m = R.shape[0]
        n = R.shape[1]
        Q = np.eye(m)
        
        for i in range(n):
            x = R[i:m, i]
            x = x.reshape(x.shape[0],1)
            
            v_i = np.zeros(x.shape)
            if (x[0] == 0):
                v_i[0] = np.linalg.norm(x)    
            else:
                v_i[0] = np.sign(x[0]) * np.linalg.norm(x)

            v_i = v_i + x

            v_i = self.normalize(v_i)

            R[i:m, i:n] = R[i:m, i:n] - 2 * v_i.dot(v_i.T.dot(R[i:m, i:n]))
            Q[:,i:] = Q[:,i:]-(Q[:,i:].dot(v_i)).dot(2*v_i.T)
            
        return Q,R
    
    def QR_linear_solver(self):
        assert(self.A.shape[0] == self.A.shape[1])
        n = self.A.shape[1]
        
        Q,R = self.QR_factorization()
        right = Q.T.dot(self.b)
        solution = np.ones((n, 1), dtype='float')
        
        for i in range(n)[::-1]:
            r_i = R[i,:]
            coeff = r_i[i]
            r_i[i] = 0
            left = r_i.dot(solution)
            solution[i] = (right[i] - left)/coeff
            
        return solution
    
    def classic_linear_solver(self):
        assert(self.A.shape[0]==self.A.shape[1])
        return np.linalg.inv(self.A).dot(self.b)
    
    @staticmethod
    def normalize(vect):
        return vect/np.linalg.norm(vect)
    
    def linear_solver_comparaison(self, maxN=30):
        size=[]
        classic_time=[]
        qr_time=[]
        for i in range(maxN):
            self.load_new(A=np.random.random((i,i)), b=np.random.random((i,1)))
            
            size.append(i)
            
            start_time1 = time.time()
            self.classic_linear_solver()
            stop_time1 = time.time()
            classic_time.append(stop_time1-start_time1)
            
            start_time2 = time.time()
            self.QR_linear_solver()
            stop_time2 = time.time()
            qr_time.append(stop_time2-start_time2)
            
        df = pd.DataFrame({'Classic':classic_time, 'QR':qr_time}, index=size)
        print(df.tail())
        df.to_csv('classic_vs_qr.csv')
        # plt.scatter(df.index, df['Classic'], color = 'red')
        # plt.scatter(df.index, df['QR'], color = 'blue')
        df.plot(kind='line', subplots=False)
        plt.xlabel('Size of matrix A')
        plt.ylabel('Time in (seconds)')
        plt.title('Classic vs QR solver')
        plt.savefig('cls_vs_qr.png')
        
    def QR_LSE_solver(self):
        m = self.A.shape[0]
        n = self.A.shape[1]
        Q,R = self.QR_factorization()
        right = Q.T.dot(self.b)

        alfa = right
        alfa[n:] = 0
        # beta = right[n+1:]
        
        self.load_new(A=R[:n,:], b=alfa[:n])
        return(self.classic_linear_solver())
    
    def QR_LSE_plot(self, maxN=50):
        size=[]
        time_reg=[]
        for i in range(maxN):
            self.load_new(A=np.random.random((i,i)), b=np.random.random((i,1)))
            
            size.append(i)
            
            start_time1 = time.time()
            self.classic_linear_solver()
            stop_time1 = time.time()
            time_reg.append(stop_time1-start_time1)
            if((i>0) and ((stop_time1-start_time1)>0)):
                print(stop_time1-start_time1)
                self.n_t.append(np.log(stop_time1-start_time1)/np.log(i))


        df = pd.DataFrame({'QR_LSE':time_reg}, index=size)
        # print(df.tail())
        df.to_csv('qr_lse.csv')
        # # plt.scatter(df.index, df['Classic'], color = 'red')
        # # plt.scatter(df.index, df['QR'], color = 'blue')
        # df.plot(kind='line', subplots=False)
        # plt.xlabel('Size of matrix A')
        # plt.ylabel('Time in (seconds)')
        # plt.title('LSE QR solver')
        # plt.savefig('qr_lse.png')
            
            
# Test cases
A = np.array([[1, 2, 3],
              [1, -1, 6],
              [3, 8, 2],
              [1, 5, 6]], dtype='float')
b = np.array([[1],
              [2],
              [3],
              [4]], dtype='float')

q = QRsolver(A, b)

