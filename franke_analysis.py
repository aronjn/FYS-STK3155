import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



from ML_landscape import Landscape

np.random.seed(42)


def FrankesFunction(x, y):
    """
    Franke's function has two Gaussian peaks of different
    heights, and a smaller dip. It is used as a test function in
    interpolation problems.
    The function is evaluated on the square x_i ∈ [0, 1], for all i = 1, 2.
    Available here as an example.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def plot3d(fig, ax, x, y, z, scatter=False):
    ###plot the surface
    if scatter:
        surf = ax.scatter(x, y, z, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False)
    else:
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ###add a color bad which maps value to colors
        fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('z', fontsize=16)



###generate data set
N = 51  #nr of samples/data points
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
x = np.sort(x)
y = np.sort(y)
XX, YY = np.meshgrid(x,y)       #matrix form
ZZ = FrankesFunction(XX, YY)  #matrix form
ZZ += 0.1*np.random.normal(0, 1, size=(XX.shape)) #added noise
z = np.ravel(ZZ)   #vector form

"""
###plotting data set
fig = plt.figure()
ax = fig.gca(projection='3d')
plot3d(fig, ax, XX, YY, ZZ, scatter=True)
plt.show()
"""


A = Landscape(ZZ)
###this is part a)
###testing OLS for polynomial degree 0 to 5

"""
for deg in range(6):
    X = A.CreateDesignMatrix_X(deg)  #5th degree polynomial
    beta = A.OLS(X, z, SVD = True)

    z_tilde = X.dot(beta)
    ZZ_tilde = np.reshape(z_tilde, (N,N))  #matrix form of z_tilde

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    plot3d(fig, ax1, XX, YY, ZZ_tilde, scatter=False)  #model
    plot3d(fig, ax2, XX, YY, ZZ, scatter=True)         #dataset
    ax1.set_title("polynomial degree = %d" % deg)
    plt.show()

    print("\npolynomial degree %d" % deg)
    print("MSE = %g" % (A.MSE(z, z_tilde)))
    print("R2  = %g" % (A.R2(z, z_tilde)))


###confidence interval for degree = 5
X = A.CreateDesignMatrix_X(5)  #5th degree polynomial
beta = A.OLS(X, z, SVD = True)
z_tilde = X.dot(beta)
beta_var = A.Var(X, z, beta, z_tilde)
A.confidence_interval(beta, beta_var)
"""


###this is part b)
###we test with degree 5
"""
X = A.CreateDesignMatrix_X(5)  #5th degree polynomial
beta = A.OLS(X, z, SVD = True)
z_tilde = X.dot(beta)
MSE_train, MSE_test = A.train_test(X, 100)
print("MSE test  (train-test split) = %7g" % MSE_test)
print("MSE train (train-test split) = %7g" % MSE_train)
print("MSE (100%% train)             = %7g" % A.MSE(z, z_tilde))

MSE_train_k, MSE_test_k = A.k_fold_cross_validation(X, z, lam = 1, k=10)
print("\nMSE test  (train-test split) = %7g" % MSE_test_k)
print("MSE train (train-test split) = %7g" % MSE_train_k)
"""



###this is part c)
degrees = np.arange(30)

MSE_list_train = np.zeros(len(degrees))
MSE_list_test = np.zeros(len(degrees))
for i in degrees:
    print(i)
    X = A.CreateDesignMatrix_X(deg = i)
    MSE_train, MSE_test = A.train_test(X, 100)
    MSE_list_train[i] = MSE_train
    MSE_list_test[i] = MSE_test


plt.plot(MSE_list_train, label='train')
plt.plot(MSE_list_test, label='test')
plt.xlabel('Complexity', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.legend()
plt.show()




"""
lambda_list = np.arange(0.02, 0.3, 0.02)
deg_list = range(50)

AA = len(deg_list)
BB = len(lambda_list)
MSE_values_test = np.zeros((BB, AA))
MSE_values_train = np.zeros((BB, AA))
#print(MSE_values_test.shape)
#MSE_values_test = np.zeros(len(deg_list))

#print(X_train)
#print('------------------------------')    #print(X_train)
"""

"""
for i in range(len(lambda_list)):
    lam = lambda_list[i]
    print(i/BB)
    for j in deg_list:
        X = A.CreateDesignMatrix_X(XX, YY, deg = j)  #5th degree polynomial
        MSE_train, MSE_test = A.k_fold_cross_validation(X, z, lam)

        MSE_values_test[i, j] = MSE_test
"""



"""
plt.plot(MSE_values_train, label='train', marker='o')
plt.plot(MSE_values_test, label='test', marker='o')
plt.legend()
"""

"""
plt.imshow(MSE_values_test, origin='lower')
plt.colorbar()
plt.xlabel('degrees')
plt.ylabel('lambda')
plt.show()
"""


"""
plt.plot(MSE_values_test[0,:])
plt.plot(MSE_values_train[0,:])
plt.show()
"""
