import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.linalg import svd, diagsvd, pinv
###
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sklearn as skl



class Landscape:
    """
    This class models a dataset on the form z = f(x, y), hence the name 'Landscape'.
    It uses the methods Ordinary Least Squares (OLS), Ridge, and LASSO to make
    models to fit the dataset.
    It also uses the resampling techniques train-data split and k-fold cross validation.
    The model used is polynomials in x and y.
    """

    def __init__(self, ZZ):
        self.ZZ = ZZ          #data set on grid form [y, x]
        self.z = np.ravel(ZZ)  #data set as flat array

        self.Nx = len(ZZ[0,:])
        self.Ny = len(ZZ[:,0])

        self.x = np.linspace(ZZ[0,0] , ZZ[0,-1], self.Nx)
        self.y = np.linspace(ZZ[0,0] , ZZ[-1,0], self.Ny)

        XX, YY = np.meshgrid(self.x, self.y)
        self.XX = XX
        self.YY = YY


    def CreateDesignMatrix_X(self, deg):
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword agruments deg is the degree of the polynomial you want to fit.
        """
        x = self.XX
        y = self.YY

        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((deg+1)*(deg+2)/2)		# Number of elements in beta
        X = np.ones((N,l))

        for i in range(1,deg+1):
        	q = int((i)*(i+1)/2)
        	for k in range(i+1):
        		X[:,q+k] = x**(i-k) * y**k

        return X

    def OLS(self, X, z, SVD = False):
        """
        Ordinary Least Squares (OLS). It estimates a beta.
        It uses either matrix inversion or Singular Value Decomposition (SVD).
        """
        #z = self.z

        ###making sure that z is a flat array
        if len(z.shape) > 1:
            z = np.ravel(z)

        if SVD:
            U, d, Vh = svd(X, full_matrices = False)
            beta = np.dot(Vh.T, U.T @ z.T/d)

        else:
            beta = np.linalg.inv( X.T.dot(X)).dot(X.T).dot(z)

        return beta


    def ridge(self, X,   lam, SVD = False):
        """
        Ridge. It estimates a beta.
        It uses either matrix inversion or Singular Value Decomposition (SVD).
        """
        z = self.z
        ###making sure that z is a flat array
        if len(z.shape) > 1:
            z = np.ravel(z)


        if SVD:
            U, d, Vh = svd(X)
            beta = Vh.T @ pinv(diagsvd(d, U.shape[0], Vh.shape[0])) @ U.T @ z

        else:
            p = len(X[0,:])
            beta = np.linalg.inv( X.T.dot(X) + lam*np.identity(p) ).dot(X.T).dot(z)

        return beta


    def LASSO(self, X, lam):
        """
        doc string
        """
        z=self.z

        ###making sure that z is a flat array
        if len(z.shape) > 1:
            z = np.ravel(z)

        clf_lasso = Lasso(alpha = lam, fit_intercept=False, max_iter = 10e5).fit(X, z)
        beta = clf_lasso.coef_
        #print(beta)

        return beta


    def Var(self, X, z, beta, z_tilde):
        """
        Computes the variance of the estimated beta.
        """
        #z = self.z
        N = len(z)
        p = len(X[0,:])
        sigma = (1/(N - p - 1))*np.sum((z - z_tilde)**2)
        #print(sigma)
        VAR = np.linalg.inv(X.T.dot(X))*sigma**2
        return VAR.diagonal()


    def MSE(self, data, model):
        data = np.ravel(data)
        model = np.ravel(model)
        #n = len(data)
        return np.mean((data - model)**2)

    def R2(self, data, model):
        n = len(data)
        A = np.sum((data - model)**2)
        y_mean = np.mean(data)
        B = np.sum((data - y_mean)**2)
        return (1 - (A/B))


    def k_fold_cross_validation(self, X, z, lam = 1, k=10):
        N = int(len(X[:,0])/k)  #length of folder
        p = 0  #starting value for lower bound
        #np.random.shuffle(X)  #shuffle the rows of X
        MSE_values_test_k  = np.zeros(k)
        MSE_values_train_k = np.zeros(k)
        for i in range(k):
            p = int(p)

            X_k_test = X[p:p+N, :]

            index = range(p, p+N)
            X_k_train = np.delete(X, index, axis=0)

            z_k_test = z[p:p+N]
            z_k_train = np.delete(z, index)

            #beta_train = self.ridge(X_k_train, z_k_train, lam)
            beta_train = self.OLS(X_k_train, z_k_train, SVD = True)

            z_tilde_train = X_k_train.dot(beta_train)
            z_tilde_test = X_k_test.dot(beta_train)

            MSE_values_train_k[i] = self.MSE(z_k_train, z_tilde_train)
            MSE_values_test_k[i] = self.MSE(z_k_test, z_tilde_test)
            p += N
        #print(MSE_values)
        #print(np.mean(MSE_values))
        return np.mean(MSE_values_train_k), np.mean(MSE_values_test_k)


    def train_test(self, X, N):
        """
        This function splits the data into training and test splits N times
        and then takes the average of the MSE and returns it.
        """
        z = self.z
        MSE_train = np.zeros(N)
        MSE_test = np.zeros(N)
        for i in range(N):
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
            beta_train = self.OLS(X_train, z_train, SVD = True)
            z_tilde_train = X_train.dot(beta_train)
            z_tilde_test = X_test.dot(beta_train)
            MSE_train[i] = self.MSE(z_train, z_tilde_train)
            MSE_test[i] = self.MSE(z_test, z_tilde_test)
        G_train = np.mean(MSE_train)
        G_test = np.mean(MSE_test)
        return G_train, G_test





    def FrankesFunction(self, x, y):
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

    def plot3d(self, fig, ax, x, y, z, scatter=False):
        ###plot the surface
        if scatter:
            surf = ax.scatter(x, y, z, cmap=cm.coolwarm,
                                  linewidth=0, antialiased=False)
        else:
            surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            ###add a color bad which maps value to colors
            fig.colorbar(surf, shrink=0.5, aspect=5)

        ###customize the z axis
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #ax.set_xlabel('x', fontsize=16)
        #ax.set_ylabel('y', fontsize=16)
        #ax.set_zlabel('z = f(x,y)', fontsize=16)

    def confidence_interval(self, beta, beta_var):
        beta_CI_diff = 1.96*np.sqrt(beta_var)

        print('            value      95% confidence')
        for i in range(len(beta)):
            print("beta_%-2s = %8g +- %-10g" % (i, beta[i], beta_CI_diff[i]))



if __name__ == "__main__":
    None
