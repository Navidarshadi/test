#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@





# """
# Created on Wed Dec 30 17:16:13 2020

# @author: navidarshadi
# """

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sun Dec 20 06:24:47 2020

# @author: navidarshadi
# """

##Generating multivariant-normal     [X1 X2...XN] , size => (k*N)
#Cov and mean
import numpy as np
import numpy,scipy

N = 8   #number of original data   [X1 X2 X3 X4] 
k = 2  #dimension of original data  
cov = np.zeros((k, k), int)
np.fill_diagonal(cov, 1)
print('cov=', cov )

#generating mean which must be k length (here zeros)
mean = np.zeros(k)
print('mean = ', mean)
print()

#[X1 X2...XN]:    
import numpy as np


Gt = np.random.multivariate_normal(mean,cov,size=N)
#mean  : 1-D array_like, of length N Mean of the N-dimensional distribution. for k=2 = ([0,0]) . 
#cov   : 2-D array_like, of shape (N, N). for k=2 => [[1,0,],[0,1]] 
print('Gt,shape=',Gt.shape)

print('Gt=') ,print(Gt)


G=Gt.transpose()
print('G,shape=',G.shape) 
print('Gussian Matrix G ='),print(G)
##plotting 2 first columns of G (or full picture in 2 dimensions case)
#-for N high shows gaussian behaviour  '
print('2 first columns of G ')
print(G[0,:],G[1,:])
#ax = plt.scatter(G[0,:],G[1,:])


print('M=number of connected neighbour')
M = N-1
print(M)

from sklearn.neighbors import kneighbors_graph
X = Gt #X must be of shape of (N,K).  (number of samples, number of features)=(number of points, Dimension)=(N,k) 
#so becuase  our gaussian shape is (K,N) we transpose it to use bellow function 

##composing distance function from different input vectors(nodes) which have gaussian distributions:
A = kneighbors_graph(X, M , mode='distance', metric='minkowski',p=2, include_self=False) ## mode= 'connectivity' or 'distance'
# N data points are connected to M = N-1 rest of data points and their euclidean distances are the elements of the D matrix
#therefor we end up with higher dimension matrix (N,K) to (N,N) after graph learning because MDS algorthm 
#print(A)
Dsqrt = np.array(A.toarray())
#print(Dsqrt)
#since in the paper is mentioned elements of distance matrix D (i.e dij) must be squar elements of distance matrix using Euclidian distance 
D = np.square(Dsqrt)
print ('D = weight matrix: ')
#print(Dij[2,0])
print('D shape=(N,N)=')
print(D.shape)
print('D=')
print(D)

#.picking elemnts of the weight matrix with i and j indexes :
#i = 1 
#j = 2
#print('for i and j =') , print(i), print('&') , print (j)
#dij = D[i,j] 
#print('dij=')
#print(dij)

## MultiDimensional Scaling (MDS) 

# (1) composing  L = In − (1/n)1n1Tn 
In = np.identity(N) 
one = np.ones((N,1), dtype=int)  #1n
L = In - (1/N)*np.matmul(one,one.transpose())  # In − (1/N)1n1Tn
print(),print('L = ') #L must be in shape of (N,N)
print(L)
In = np.identity(N)

print(N)

# (2)  τ(D) = − 1/2 L*D*L
τ = -(1/2)*L @ D @ L
print('τ =')
print(τ)

print('SVD=>')

##calculating SVD of τ(D)

from numpy.linalg import svd

U, S, UT = svd(τ)

print("Left Singular Vectors or U :")
print(U)
print('shape of U')
print(U.shape)
print("Singular Values Σpower :") 
print(np.diag(S))
Σpower = np.diag(S)
print('Σpower'),print(Σpower)
print('shape of Σpower')
print(Σpower.shape)
print("Right Singular Vectors or UT:"),print(UT) 



##τ2(D) = U2(Σ^2)UT2  which is the Best rank 2 approximation of τ(D): 
#approx because we have 𝞂1>𝞂2>... the columns and rows corresponded
#-to 𝞂1 are more important than 𝞂2 and so on! in best rank 2 approx.
#-we tak only 𝞂1 and 𝞂2 
Σpowerrank2= Σpower[0:2,0:2]
print('Σpowerrank2') ,print(Σpowerrank2) #must have size of (2,2) 
## 

U2 = U[0:N,0:2]
print('U2'),print(U2)
print('U2T = '),print(UT[0:2,0:N])
print('Σpowerrank2'),print(Σpowerrank2)
Σrank2 = np.sqrt(Σpowerrank2)
print('Σrank2 ='),print(Σrank2)

# Xappr approximation:   Xappr = U2 * Σrank2.
Xappr = np.matmul(U2,Σrank2)
print('G = '),print(G)
print('X='),print(X) 
print('Xapproximation='),print(Xappr)
import matplotlib.pyplot as plt
##plot to compair the original data vs the approximated data
#-(but only for original data 2D can be visualized  )
plt.scatter(G[0,:],G[1,:])
plt.scatter(Xappr[:,0],Xappr[:,1])
plt.show()
# in case of dimension=2 we can obviously see in the approximation
#-is only in rotation,refletion differ.


###calculate distortion

##Appending columns of zeros to approx result

#to measure the distortion the resulting space must be of the same 
#-dimensionality as the original space.we can always append columns
#- of zeros to the result
print()
import numpy as np
Xappend = np.zeros(X.shape)
Xappend[:,0:2] = Xappr
print('Xappend: ',Xappend)
print(X.shape == Xappend.shape)
print()
##

## d(X,Xappend) => distortion calculation  
#for i in range (1 , 2): 
          

def R2d(i):
     #defining a function to generate rotation matrix (only 2D) 
     theta = np.radians(i)
     c = np.cos(theta) 
     s = np.sin(theta)
     R = np.array(((c, -s), (s, c)))  
     return(R)
# def dis(n,i):
    
    
#   dis = dis + ((numpy.linalg.norm(X[n,:] - np.matmul(R2d(i),Xappend[n,:]))))**2
#   return(dis) 

#print('now'),print(dis)
# i=1
# n=1


def minrotation(X,Xappend):
    maxrotation=10
    distorsion = np.zeros((maxrotation, 1))
    #print('generating distortion matrix space with zeroes elements'), print(distorsion)
    
    
    #print('X[0,:]='),print(X[0,:]),print('X[1,:]='),print(X[1,:])
    for i in range(0,maxrotation):  
        dis = 0
        for n in range(0,N):
              # print('R*Xappend='), print(np.matmul(R2d(i),Xappend[n,:]))
              Z = X[n,:] - np.matmul(R2d(i),Xappend[n,:])
              dis = dis + (numpy.linalg.norm(Z))**2
              #print('for i='),print(i),print('and for n='),print(n)
              print('Xappend[n,:]'),print(Xappend[n,:]),print('R2d(i)'),print(R2d(i)),print('R2d(i)@Xappend[n,:]'),print(R2d(i)@Xappend[n,:]),print('Z ='),print(Z),print('dis'),print(dis)
              # print('for i='),print(i),print('and for n='),print(n) 
        #print('dis(i,n)=') ,print(dis)  
        #print('dis matrix='),
        distorsion[i] = dis
    print(distorsion)
    print('min(distorsion)=')
    
    print(min(distorsion))
    return(min(distorsion))
    
minrotation(X,Xappend)

    
          
  
#        R2d(i) 
#        xn1 = X[n,:] #vector(takes the nth rows of original data matrix X  )
#        xn2 = Xappend[n,:] #vector(takes the nth rows of approximated appended data matrix Xappend  )
#        dis = 0
#        dis = dis + ((numpy.linalg.norm(xn1 - np.matmul(R2d(i),xn2) )))**2
# #      Rxn2 = np.matmul(R2d(i),xn2)
#      # print('Rxn2'),print(xn2)
#       dis = dis + ((numpy.linalg.norm(xn1 - Rxn2 )))**2
#       print('dis = '),print(dis)
  
# #print('dis = '),print(dis)
# print()
# print('dis final'),print(dis)
# # print('xn1'),print(xn1)
# # print('xn2'),print(xn2)

# # ## d(X,Xappend)=




###number of non-zero elements in the distance matrix and 
#-precision in reconstruction approximation:
#where only a portion of the entries in the distance matrix are available.
print('D='),print(D)
Dnum = numpy.count_nonzero(D)   
print('number of non-zero elements in D=',Dnum)
print('8n = ') ,print(8*N)
if Dnum>8*N :
  print('num of distance elements in D is not sufficient for precise reconstruction')
else:
    print('sufficient num of distance elements in D for precise reconstruction')

    


