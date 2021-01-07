# Xapprox rotation & deterministic


# deterministic input 


"""
first test
Created on Sat Jan  2 19:42:58 2021

@author: navidarshadi
"""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

k =2
import numpy as np
import numpy,scipy
X = np.array([[ 3, 0],[ 4, 4],[ 4, 3],[ 9, 4],[ 4, 9],[ 7, 1],[ 1, 4],[ 6, 3],[ 3, 3],[ 3, 0],[ 3, 0],[ 4, 4],[ 9, 3],[ 1, 4],[ 0, 9],[ 7, 8],[ 1, 2],[ 6, 3],[ 9, 3],[ 3, 0],[ 3, 0],[ 4, 4],[ 4, 6],[ 0, 4],[ 4, 2],[ 2, 1],[ 3, 4],[ 1, 3],[ 1, 3],[ 3, 9],[ 3, 0],[ 4, 4],[ 4, 3],[ 9, 4],[ 4, 9],[ 7, 1],[ 1, 4],[ 6, 3],[ 3, 3],[ 3, 0],[ 3, 0],[ 4, 4],[ 4, 1],[ 9, 3],[ 4, 4],[ 7, 5],[ 5, 4],[ 6, 5],[ 3, 5],[ 3, 5]])
#X = np.array([[ 3, 0],[ 4, 4],[ 4, 3],[ 9, 4],[ 4, 9],[ 7, 1]])
X = np.array([[ 3, 0],[ 4, 4],[ 4, 3],[ 9, 4],[ 4, 9],[ 7, 1],[ 3, 0],[ 4, 4],[ 4, 3],[ 9, 4],[ 4, 9],[ 7, 1],[ 3, 0],[ 4, 4],[ 4, 3],[ 9, 4],[ 4, 9],[ 7, 1]])
#X =X-np.mean(X,axis = 0)
print('X.mean'),print(X.mean)
print('X=',X)
b = X.shape
print(b)
N = b[0]
print(N)
k =2


print('X,shape=',X.shape)
print('2 first columns of X ')
print(X[:,0],X[:,1])
print('M=number of connected neighbour')
M = N-1
print(M)

from sklearn.neighbors import kneighbors_graph
 #X must be of shape of (N,K).  (number of samples, number of features)=(number of points, Dimension)=(N,k) 
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

#1/2 L*D*L
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
print('shape of Σpower')
print(Σpower.shape)
print("Right Singular Vectors or UT:"),print(UT) 

#test τ = U@Σpower@UT



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
Σrank2 = np.sqrt(Σpowerrank2)
print('Σrank2 ='),print(Σrank2)

# Xappr approximation:   Xappr = U2 * Σrank2.
Xappr = np.matmul(U2,Σrank2)
print('X = '),print(X)
print('X='),print(X) 
print('Xapproximation='),print(Xappr)
import matplotlib.pyplot as plt
##plot to compair the original data vs the approximated data
#-(but only for original data 2D can be visualized  )

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
print('Xappend: '),print(Xappend)
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



print('Xappend1',Xappend)
Xappend = Xappend@np.diag([-1,1])
print('Xappend2',Xappend)
#Xappend = Xappend@np.diag([-1,1])
maxrotation=360
distorsion = np.zeros((maxrotation, 1))       
for i in range(0,maxrotation):  
    dis = 0
    for n in range(0,N):
          # print('R*Xappend='), print(np.matmul(R2d(i),Xappend[n,:]))
          Z = X[n,:] - np.matmul(R2d(i),Xappend[n,:])
          dis = dis + (numpy.linalg.norm(Z))**2
          #print('for i='),print(i),print('and for n='),print(n)
          #print('Xappend[n,:]'),print(Xappend[n,:]),print('R2d(i)'),print(R2d(i)),print('R2d(i)@Xappend[n,:]'),print(R2d(i)@Xappend[n,:]),print('Z ='),print(Z),print('dis'),print(dis)
          # print('for i='),print(i),print('and for n='),print(n) 
    #print('dis(i,n)=') ,print(dis)  
    #print('dis matrix='),
    distorsion[i] = dis
print(distorsion)
print('min(distorsion)=')
print(min(distorsion))



result = np.argmin(distorsion) #the index of min
print('Tuple of arrays returned : ', result) # shows which rotation gives closest distance



Xrotated = Xappend@R2d(result)
print(R2d(result))
print(Xrotated)
plt.scatter(Xrotated[:,0],Xrotated[:,1],color='red')
plt.scatter(X[:,0],X[:,1],color='black') 
#plt.scatter(Xappend[:,0],Xappend[:,1])

plt.show()







# print('dis?'), print(dis)

