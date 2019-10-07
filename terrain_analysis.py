from imageio import imread
import numpy as np
import matplotlib.pyplot as plt

from ML_landscape import Landscape

terrain1 = np.array(imread("SRTM_data_Norway_1.tif"))
ZZ = terrain1[:50, :50]

Nx = len(ZZ[0,:])
Ny = len(ZZ[:,0])

x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
XX, YY = np.meshgrid(x, y)

A = Landscape(ZZ)

"""
X = A.CreateDesignMatrix_X(deg = 7)
beta = A.OLS(X, SVD = True)
z_tilde = X @ beta
ZZ_tilde = np.reshape(z_tilde, (Ny,Nx))  #matrix form of z_tilde

fig = plt.figure(figsize=plt.figaspect(0.5))
#ax1 = fig.add_subplot(1,2,1, projection = '3d')
#ax2 = fig.add_subplot(1,2,2, projection = '3d')
#ax1.plot_surface(XX, YY, ZZ)
#ax2.plot_surface(XX, YY, ZZ_tilde)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.imshow(ZZ)
ax2.imshow(ZZ_tilde)


ax1.imshow(ZZ)
ax2.imshow(ZZ_tilde)


plt.show()
"""

z = np.ravel(ZZ)


for deg in range(6):

    X = A.CreateDesignMatrix_X(deg)  #5th degree polynomial
    beta = A.OLS(X, SVD = True)
    #beta = A.ridge(X, 0.001, SVD = True)
    beta=A.LASSO(X, lam=0.001)

    z_tilde = X.dot(beta)
    ZZ_tilde = np.reshape(z_tilde, (Ny,Nx   ))  #matrix form of z_tilde

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.imshow(ZZ, origin='lower')
    ax2.imshow(ZZ_tilde, origin='lower')
    ax1.set_title("polynomial degree = %d" % deg)
    plt.show()

    print("\npolynomial degree %d" % deg)
    print("MSE = %g" % (A.MSE(z, z_tilde)))
    print("R2  = %g" % (A.R2(z, z_tilde)))
