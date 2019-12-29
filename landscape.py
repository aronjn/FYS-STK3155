"""
this is a program that performs regression on a "landscape function" on the form
z=f(x,y).
it will approximate z using polynomials of x and y.
the main program does this on the franke function, but it is intended to be
applied on a general dataset on that form.

notation:
x = flat array
XX = grid array
"""

import numpy as np
seed = 10
np.random.seed(seed)

import sys

from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model    import Lasso

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib           import cm
from matplotlib.ticker    import LinearLocator, FormatStrFormatter

class landscape:
    def __init__(self, ZZ):
        self.ZZ = ZZ

    def create_design_matrix(self, x, y, n = 5):
    	"""
    	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    	"""
    	if len(x.shape) > 1:
    		x = np.ravel(x)
    		y = np.ravel(y)

    	N = len(x)
    	l = int((n+1)*(n+2)/2)		# Number of elements in beta
    	X = np.ones((N,l))

    	for i in range(1,n+1):
    		q = int((i)*(i+1)/2)
    		for k in range(i+1):
    			X[:,q+k] = x**(i-k) * y**k

    	return X


    def OLS(self, z, X):
        beta = np.linalg.inv(X.T@X)@(X.T@z)
        return beta

    def ridge(self, z, X, lam):
        p = len(X[0,:])
        #beta = np.linalg.inv( X.T.dot(X) + lam*np.identity(p) ).dot(X.T).dot(z)
        beta = np.linalg.inv(X.T@X + lam*np.identity(p))@(X.T@z)
        return beta


    def MSE(self, z, z_hat):
        error = np.mean((z - z_hat)**2)
        return error

    def R2(self, z, z_hat):
        A = np.sum((z - z_hat)**2)
        B = np.sum((z - np.mean(z))**2)
        C = 1 - (A/B)
        return C

    def conf_int(self, z, z_hat, X, beta):
        ### computes the variance of the beta
        N = len(z)
        p = len(X[0,:])
        sigma = (1/(N - p - 1))*np.sum((z - z_hat)**2)
        VAR = np.linalg.inv(X.T@X)*sigma**2
        VAR = VAR.diagonal()
        beta_CI = 1.96*np.sqrt(VAR)
        print("\n            value     95% confidence")
        for i in range(len(beta)):
            #print(beta_CI[i])
            print("beta %2d = %8g +- %10g" % (i, beta[i], beta_CI[i]))

    def plot_landscape(self, ZZ, title=None, scatter=False, save=False, name=None):
        ZZ_real = self.ZZ
        vmin = np.min(ZZ_real)
        vmax = np.max(ZZ_real)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        x = np.linspace(0,1, len(ZZ))
        y = np.linspace(0,1, len(ZZ))
        XX, YY = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if scatter:
            surf = ax.scatter(XX, YY, ZZ, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)
        else:
            surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm,
                                   linewidth=0, norm=norm, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlabel('z', fontsize=16)
        ax.set_title(title, fontsize=16)

        if save:
            plt.savefig(name)



        plt.show()


if __name__ == "__main__":
    #print('hello')

    def FrankeFunction(x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    N = 100  #nr of points in x and y direction
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    XX, YY = np.meshgrid(x, y)
    ZZ = FrankeFunction(XX, YY)
    z = np.ravel(ZZ)

    ### adding noise
    noise = 0.1
    ZZ += noise*np.random.normal(loc=0, scale=1, size=(ZZ.shape))

    if len(sys.argv) < 2:
        print("add argument")
        sys.exit
    else:
        exer_input = str(sys.argv[1])

    if exer_input == "-a":

        ### using instance
        A = landscape(ZZ)
        A.plot_landscape(ZZ, save=True, name="original_noise", title=r"Noise = $0.1 \times N(0,1)$")
        #A.plot_landscape(ZZ, save=True, name="original_no_noise", title=r"No noise")

        ### making the desing matrix
        #print(X)

        ### OLS

        ### checking polynomial degrees
        for deg in range(6):
            X = A.create_design_matrix(XX, YY, n=deg)
            beta = A.OLS(z, X)
            z_hat = X@beta
            ZZ_hat = z_hat.reshape(XX.shape)
            print("\nMSE (deg=%2d) = %10g" % (deg, A.MSE(z, z_hat)))
            print("R2  (deg=%2d) = %10g" % (deg, A.R2(z, z_hat)))
            #MSE_array[deg] = A.MSE(z, z_hat)

            A.plot_landscape(ZZ_hat, title="Degree %d" % deg, save=True, name="OLS_deg%d" % deg)

        X = A.create_design_matrix(XX, YY, n=5)
        beta = A.OLS(z, X)
        A.conf_int(z, z_hat, X, beta)

    elif exer_input == "-b":
        B = landscape(ZZ)
        X = B.create_design_matrix(XX, YY, n=5  )
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.30, random_state=seed)

        ### checking polynomial degrees with train-test-split
        print("TRAIN-TEST SPLIT")
        MSE_array_no = np.zeros(6)
        MSE_array_tts = np.zeros(6)
        MSE_array_k = np.zeros(6)
        for deg in range(6):
            X = B.create_design_matrix(XX, YY, n=deg)

            #train-test split
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.30, random_state=seed)
            beta = B.OLS(z_train, X_train)
            z_hat = X_test@beta
            MSE_array_tts[deg] = B.MSE(z_test, z_hat)

            #no resampling
            beta_no = B.OLS(z, X)  #used for comparison
            z_hat_no = X@beta
            MSE_array_no[deg] = B.MSE(z, z_hat_no)

            kf = KFold(n_splits=10, random_state=None, shuffle=True)
            MSE_array_mean = np.zeros(10)
            R2_array_mean = np.zeros(10)
            i = 0
            for train_index, test_index in kf.split(X):
                #print("Train:", train_index, "Validation:",test_index)
                X_train, X_test = X[train_index], X[test_index]
                z_train, z_test = z[train_index], z[test_index]

                beta = B.OLS(z_train, X_train)
                z_hat = X_test@beta
                MSE_array_mean[i] = B.MSE(z_test, z_hat)
                R2_array_mean[i] = B.R2(z_test, z_hat)
                i += 1
            MSE_array_mean = np.mean(MSE_array_mean)
            MSE_array_k[deg] = MSE_array_mean


        plt.plot(MSE_array_no, marker='o', label=r'no resampling')
        plt.plot(MSE_array_tts, marker='o', label=r'train-test split')
        plt.plot(MSE_array_k, marker='o', label=r'k-fold cross validation')
        print(np.min(MSE_array_k))
        plt.ylabel(r"MSE")
        plt.legend()
        plt.xlabel(r"polynomial degrees")
        plt.show()

    elif exer_input == "-d":
        D = landscape(ZZ)

        ### k-fold cross validation
        ### determine lambda
        lambda_vals = np.logspace(-4,1,10)
        MSE_array_k = np.zeros([6, 10])
        for k in range(len(lambda_vals)):
            for deg in range(6):
                X = D.create_design_matrix(XX, YY, n=deg)
                kf = KFold(n_splits=10, random_state=None, shuffle=True)
                MSE_array_mean = np.zeros(10)
                i = 0
                for train_index, test_index in kf.split(X):
                    #print("Train:", train_index, "Validation:",test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    z_train, z_test = z[train_index], z[test_index]

                    #print(lambda_vals[k])
                    beta = D.ridge(z_train, X_train, lam=lambda_vals[k])
                    #print("\n")
                    #print(beta)
                    #print(D.OLS(z_train, X_train, ))
                    z_hat = X_test@beta
                    MSE_array_mean[i] = D.MSE(z_test, z_hat)
                    i += 1
                MSE_array_mean = np.mean(MSE_array_mean)
                #print(MSE_array_mean)
                MSE_array_k[deg][k] = MSE_array_mean

        plt.imshow(MSE_array_k, origin="lower", extent=[np.min(lambda_vals)*1, np.max(lambda_vals)*1, 0, 5])
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"degree")
        plt.colorbar()
        plt.title(r"MSE")
        plt.show()
        print(np.where(MSE_array_k == np.min(MSE_array_k)))
        print(MSE_array_k[5][0])
        print(lambda_vals[0])

    elif exer_input == "-e":
        E = landscape(ZZ)

        ### k-fold cross validation
        ### determine lambda
        alpha_vals = np.logspace(-2,-1,10)
        MSE_array_k = np.zeros([6, 10])
        for k in range(len(alpha_vals)):
            for deg in range(6):
                X = E.create_design_matrix(XX, YY, n=deg)
                kf = KFold(n_splits=10, random_state=None, shuffle=True)
                MSE_array_mean = np.zeros(10)
                i = 0
                for train_index, test_index in kf.split(X):
                    #print("Train:", train_index, "Validation:",test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    z_train, z_test = z[train_index], z[test_index]

                    #print(lambda_vals[k])

                    clf = Lasso(alpha=alpha_vals[k])
                    clf.fit(X_train, z_train)
                    beta = clf.coef_
                    z_hat = X_test@beta
                    MSE_array_mean[i] = E.MSE(z_test, z_hat)
                    #print(E.MSE(z_test, z_hat))
                    i += 1
                MSE_array_mean = np.mean(MSE_array_mean)
                print(MSE_array_mean)
                MSE_array_k[deg][k] = MSE_array_mean

        plt.imshow(MSE_array_k, origin="lower", extent=[np.min(alpha_vals)*1, np.max(alpha_vals)*1, 0, 5])
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"degree")
        plt.colorbar()
        plt.title(r"MSE")
        plt.show()
        print(np.where(MSE_array_k == np.min(MSE_array_k)))
        print(MSE_array_k[4][8])
        print(alpha_vals[8])
