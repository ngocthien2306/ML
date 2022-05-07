'''
The following code is mainly from Chap 4, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb

LAST REVIEW: March 2022
'''
#Tham khảo source code của Thầy
# In[0]: IMPORTS, SETTINGS
#region
import sys
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import numpy as np
import os   
np.random.seed(42) # to output the same result across runs
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)       
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd") # Ignore useless warnings (see SciPy issue #5998)
font_size = 14
let_plot = True
#endregion


''' WEEK 07 '''

# In[1]: LINEAR REGRESSION USING NORMAL EQUATION 
# 1.1. Generate linear-looking data 
n_samples = 300
# Khởi tạo feature X cho sample
# Random thang điểm từ 0 -> 10 để dự đoán điểm số của sinh viên
X = 2*np.random.rand(n_samples, 1) # random real numbers in [0,10]
# Khởi tạo hàm số bậc nhất, để mô tả dữ liệu
y_no_noise = 3 + 7*X; 
# Random để tạo ra dữ liệu nhiễu 
y = y_no_noise + np.random.randn(n_samples, 1) # noise: random real numbers with Gaussian distribution of mean 0, variance 1

let_plot = True
if let_plot:
    plt.plot(X, y, "k.")
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.ylabel("$y$", rotation=0, fontsize=font_size)
    plt.axis([0, 5, 0, 30])
    plt.savefig("generated_data_plot",format='png', dpi=300)
    plt.show()


#%% 1.2. Compute Theta using Normal Equation 
# tạo ma trận 1 là x0 để ma trận có dạng (n + 1, n + 1)
X_add_x0 = np.c_[np.ones((n_samples, 1)), X]  # add x0 = 1 to each instance
theta_norm_eq = np.linalg.inv(X_add_x0.T @ X_add_x0) @ X_add_x0.T @ y 
# Note: Theta is a bit different from the true parameters due to the noise.
	
# 1.3. Try prediction 
X_test = np.array([[0], [2], [15]]) # 3 instances
X_test_add_x0 = np.c_[np.ones((len(X_test), 1)), X_test]  # add x0 = 1 to each instance
y_predict = X_test_add_x0 @ theta_norm_eq

# 1.4. Plot hypothesis 
if let_plot:
	plt.plot(X, y_no_noise, "g-",label="True  hypothesis")
	plt.plot(X_test, y_predict, "r-",label='Hypothesis')
	plt.plot(X, y, "k.",label='Training sample')
	plt.axis([0, 5, 0, 30])
	plt.legend()    
	plt.xlabel("$x_1$", fontsize=font_size)
	plt.ylabel("$y$", rotation=0, fontsize=font_size)
	plt.show()	


# In[2]: LINEAR REGRESSION USING GRADIENT DESCENT 

# 2.1. Gradient descent (>> see slide)

#%% 2.2. Batch gradient descent
eta = 0.1  # learning rate
m = len(X)
np.random.seed(42);
theta_random_init = np.random.randn(2,1)
theta = theta_random_init  # random initialization
#for iteration in range(1,1000) # use this if you want to stop after some no. of iterations, eg. 1000
while True:
	#gradients = 2/m * X_add_x0.T @ (X_add_x0 @ theta - y); # WARNING: @ (mat multiply) causes weird indent errors when running in Debug interactive
	gradients = 2/m * X_add_x0.T .dot (X_add_x0 .dot (theta) - y); # works the same at the code above, but no indent errors
	theta = theta - eta*gradients;
	if (np.abs(np.mean(eta*gradients)) < 0.000000001): 
		break # stop when the change of theta is small

# 2.3. Compare with theta by Normal Eq.
theta_norm_eq
theta_BGD = theta


''' WEEK 08 '''

#%% 2.4. Learning rates (>> see slide)

# 2.5. Try different learning rates 
def plot_gradient_descent(theta, eta, theta_path=None, n_iter_plot=10, n_iter_run=1000):
    m = len(X_add_x0)
    plt.plot(X, y, "k.")
    for iteration in range(1,n_iter_run): # run 1000 iter, instead of convergence stop
        if iteration <= n_iter_plot:
            y_predict = X_test_add_x0 .dot (theta) 
            if iteration == 1:
                plt.plot(X_test, y_predict, "g--", label="initial theta",linewidth=2)
            elif iteration == n_iter_plot: 
                plt.plot(X_test, y_predict, "r-", label="theta at 10th iter",linewidth=2)  
            else:
                plt.plot(X_test, y_predict, "b-",linewidth=2)                  
        gradients = 2/m * X_add_x0.T .dot (X_add_x0 .dot (theta) - y)
        theta = theta - eta*gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.axis([0, 2, 0, 15])
    plt.legend(loc='upper right');
    plt.title(r"$\eta = {}$".format(eta), fontsize=font_size)

# 2.5.1. Plot BGD  with small learning rate
np.random.seed(42)
init_theta = np.random.randn(2,1)  # random initialization
fig = plt.figure(figsize=(10,5))
plt.subplot(131);
plot_gradient_descent(init_theta, eta=0.02); plt.ylabel("$y$", fontsize=font_size)

# 2.5.2. Plot BGD with good learning rate 
plt.subplot(132); 
theta_path_bgd = [theta_random_init]
plot_gradient_descent(init_theta, eta=0.1, theta_path=theta_path_bgd)

# 2.5.3. Plot BGD with large learning rate
plt.subplot(133); 
plot_gradient_descent(init_theta, eta=0.5)
fig.suptitle("Theta values in the first 10 iterations of BGD", fontsize=14)
plt.show()


# 2.6. How to find a good learning rate eta?
#   1. Try small learning rate, then increase it gradually.
#   2. Do hyperparameter TUNING for eta (using, e.g., grid search).


''' ________________________________________________ '''


# In[3]: STOCHASTIC GRADIENT DESCENT 

# 3.0. Problems of gradient descent (>> see slide)

# 3.1. Stochastic gradient descent (>> see slide)

# 3.2. Training using Stochastic GD
def learning_schedule(t):
    alpha = 0.2; t0 = 50; # learning schedule hyperparameters
    eta = 1 / (alpha* (t + t0))
    return eta

m = len(X_add_x0)
theta = theta_random_init  # random initialization
theta_path_sgd = [theta_random_init]  

n_epochs = 50 # << 1 epoch = 1 time of running m iter (m: no. of training samples)
for epoch in range(n_epochs):
    for i in range(m):
        # Just for plotting purpose
        if epoch == 0 and i <= 20:                       
            y_predict = X_test_add_x0.dot(theta)   
            if i == 0:
                plt.plot(X_test, y_predict, "g--", label="initial theta",linewidth=2)
            elif i == 20: 
                plt.plot(X_test, y_predict, "r-", label="theta at 20th iter",linewidth=2)  
            else:
                plt.plot(X_test, y_predict, "b-",linewidth=2)     
        # Pick a random sample
        random_index = np.random.randint(m)
        xi = np.array([X_add_x0[random_index]])
        yi = np.array([y[random_index]])
        # Compute gradients
        gradients = 2 * xi.T .dot (xi .dot (theta) - yi)
        # Compute learning rate
        eta = learning_schedule(m*epoch + i)
        # Update theta
        theta = theta - eta*gradients
        theta_path_sgd.append(theta)        
plt.plot(X, y, "k.")                                 
plt.xlabel("$x_1$", fontsize=font_size)                   
plt.ylabel("$y$", fontsize=font_size)            
plt.axis([0, 2, 0, 15])  
plt.legend()
plt.title('SDG in the first 20 iter of the first epoch')
plt.show()                                            

# 3.3. Compare thetas found by SGD and BGD 
theta_BGD
theta_SGD = theta


# In[4]: MINI-BATCH GRADIENT DESCENT

# 4.1. (>> see slide)

# 4.2. Implement and run mini-batch GD
def learning_schedule(t):
    t0, t1 = 200, 1000
    return t0 / (t + t1)

n_epochs = 50
minibatch_size = 20
theta = theta_random_init  # random initialization
theta_path_mgd = [theta_random_init]
t = 0           
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_addx0_shuffled = X_add_x0[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        # Get random samples
        xi = X_addx0_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T .dot (xi .dot (theta) - yi)
        # Compute learning rate
        t += 1
        eta = learning_schedule(t)
        # Update theta
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

# 4.3. Plot update paths of BGD, SGD, and Mini-batch GB
if let_plot:
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)

    plt.figure(figsize=(7,4))
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "g-o", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "b-s", linewidth=1, label="Mini-batch")
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "r-|",markersize=7, linewidth=2, label="Batch")
    plt.legend(loc="upper left", fontsize=font_size)
    plt.xlabel(r"$\theta_0$", fontsize=font_size)
    plt.ylabel(r"$\theta_1$   ", fontsize=font_size, rotation=0)
    #plt.axis([2.5, 4.5, 2.3, 3.9])   
    plt.show()

# 4.4. Comparison of training algorithms (>> see slide) 
print("\n")


# In[5]: IMPLEMENTATION BY SCIKIT-LEARN 
# 5.1. Sklearn implementation of Normal Equation (using SVD)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# Training
lin_reg.fit(X, y.ravel()) # ravel(): convert to 1D array                  
# Learned parameters (theta)
lin_reg.intercept_, lin_reg.coef_   
# Compare with theta by previous implementation
theta_norm_eq 
# Prediction
lin_reg.predict(X_test)            

# 5.2. Sklearn implementation of Stochastic Gradient Descent 
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-9, learning_rate='optimal', alpha=0.2, random_state=42, penalty=None)
# Training
sgd_reg.fit(X, y.ravel())
# Learned parameters (theta)
sgd_reg.intercept_, sgd_reg.coef_
# Compare with theta by previous implementation
theta_SGD # different result due to no control on t0 in the learning schedule.
# Prediction
sgd_reg.predict(X_test)   


# In[6]: POLYNOMINAL REGRESSION

# 6.0. (>> see slide)

# 6.1. Generate non-linear data
m = 100
np.random.seed(30);
X = 6*np.random.rand(m, 1) - 3  # -3 < X < 3
y = 0.5*X**2 + X + 2 + np.random.randn(m, 1)
if let_plot:
    plt.plot(X, y, "k.")
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.ylabel("$y$", rotation=0, fontsize=font_size)
    #plt.axis([-3, 3, 0, 10])
    plt.title("Non-linear data")
    plt.show()

# 6.2. Add high-order features
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
# Look at a sample
X[0]           
X_poly[0]

# 6.3. Train using Normal Equation
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_  # theta

# 6.4. Train using SGD
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-9, random_state=42, penalty=None)
sgd_reg.fit(X_poly, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_ # theta

# 6.5. Prediction and plot learned models
X_test=np.linspace(-3, 3, 100).reshape(100, 1)
X_test_poly = poly_features.transform(X_test)
y_test_norm_eq = lin_reg.predict(X_test_poly)
y_test_SGD = sgd_reg.predict(X_test_poly)
if let_plot:
    plt.plot(X, y, "k.")
    plt.plot(X_test, y_test_norm_eq, "b-", linewidth=2, label="Normal Eq")
    plt.plot(X_test, y_test_SGD, "r-", linewidth=2, label="SGD")
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.ylabel("$y$", rotation=0, fontsize=font_size)
    plt.legend(loc="upper left", fontsize=font_size)
    #plt.axis([-3, 3, 0, 10])
    plt.show()


# In[7]: LEARNING CURVES

# 7.1. Try different orders of polynomial models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
for degree, plot_style in ((1, "b-"), (2, "g-"), (200, "r-")):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg)     ])
    polynomial_regression.fit(X, y)
    y_test = polynomial_regression.predict(X_test)
    plt.plot(X_test, y_test, plot_style, label="Degree = " + str(degree), linewidth=2)
plt.plot(X, y, "k.", label="Training data")
plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=font_size)
plt.ylabel("$y$", rotation=0, fontsize=font_size)
plt.axis([-3, 3, 0, 10])
plt.title("Polynomial models with various degrees")
plt.show()


# 7.2. Polynomial model of degree 200 is too complex compared to the true model! 
#   => Overfitting!
#   If you don't know the true model: How to identify overfitting? 
#   (>> see slide)

# 7.3. Plot learning curves function
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    # Split training, validation sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12)
    
    # Repeat training on m sizes of training data: time 1: use 1 sample; time 2: use 2 samples... for training
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-", linewidth=3, label="training")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="validation")
    plt.legend(loc="upper right", fontsize=font_size)    
    plt.xlabel("Training set size", fontsize=font_size)  
    plt.ylabel("Mean squared error", fontsize=font_size)   

# 7.4. Learning curve of Linear model
lin_reg = LinearRegression()
if let_plot:
    plot_learning_curves(lin_reg, X, y)
    plt.title("Learning curves of Linear model", fontsize=font_size)
    plt.axis([0, 80, 0, 2.5])   
    plt.savefig("figures/learn_curve_linear.png")
    plt.show()     

# 7.5. Explanation (>> see slide) 

# 7.6. Learning curve of Polynomial model (degree = 10)
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()) ])
if let_plot:
    plot_learning_curves(polynomial_regression, X, y)
    plt.title("Learning curves of Polynomial model (degree = 10)", fontsize=font_size)
    plt.axis([0, 80, 0, 2.5])           
    plt.savefig("figures/learn_curve_poly_10.png")
    plt.show()      

# 7.7. Explanation (>> see slide) 

# 7.8. Learning curve of Polynomial model (degree = 2)    
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin_reg", LinearRegression()) ])
if let_plot:
    plot_learning_curves(polynomial_regression, X, y)
    plt.title("Learning curves of Polynomial model (degree = 2)", fontsize=font_size)
    plt.axis([0, 80, 0, 2.5])           
    plt.savefig("figures/learn_curve_poly_2.png")
    plt.show()     


# In[8]: REGULARIZED MODELS

# 8.1. Idea (>> see slide)

# 8.2. Ridge regularization
# 8.2.1. Generate linear data
m = 20
np.random.seed(15);
X = 3*np.random.rand(m, 1)
y_no_noise = 1 + 0.5*X 
y = y_no_noise + np.random.randn(m, 1)/1.5
X_test = np.linspace(0, 3, 100).reshape(100, 1)

# 8.2.2. Train a ridge model and predict
from sklearn.linear_model import Ridge
#ridge_reg = Ridge(solver = 'cholesky', alpha=1, random_state=42) # train using closed-form solution
ridge_reg = Ridge(solver = 'sag', alpha=1, random_state=42) # train using stochastic GD
#sgd_reg = SGDRegressor(penalty="l2") # train using stochastic GD: 'l2' norm is ridge regularization
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

# 8.2.3. Plot models trained using different alphas
def plot_model(model_class, polynomial, alphas, **model_kargs):
    # Plot data and true model
    plt.plot(X, y, "k.")
    plt.plot(X, y_no_noise, 'k-', linewidth=3, label="true model")
    
    # Learn and plot trained models
    for alpha, plot_style in zip(alphas, ("g-", "b-", "r-")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model)   ])
        model.fit(X, y)
        y_test_regul = model.predict(X_test)
        plt.plot(X_test, y_test_regul, plot_style, linewidth=2, label=r"$\alpha = $" + str(alpha))
    plt.legend(loc="upper left", fontsize=font_size-1)
    plt.xlabel("$x_1$", fontsize=font_size)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(10,5))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 20, 1000), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=font_size)
plt.title("Train linear models", fontsize=font_size)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-7, 1), random_state=42)
plt.title("Train polynomial models (degree = 10)", fontsize=font_size)
plt.show()


# 8.2.4. Observation: 
#   larger alpha => stronger regularization => smaller thetas


''' ________________________________________________ '''

 

# In[9]: EARLY STOPPING

# 9.1. Idea (>> see slide) 

# 9.2. Generate non-linear data
np.random.seed(42)
m = 100
X = 6*np.random.rand(m, 1) - 3
y = 2 + X + 0.5*X**2 + np.random.randn(m, 1)
X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

# 9.3. Add high-order features and do feature scaling 
poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=70, include_bias=False)),
        ("std_scaler", StandardScaler())  ])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

# 9.4. Do early stopping 
sgd_reg = SGDRegressor(max_iter=1, tol=-np.inf, # tol<0: allow loss to increase
                       warm_start=True,   # warm_start=True: init fit() with result from previous run
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42) 
n_iter_wait = 200
minimum_val_error = np.inf  
from copy import deepcopy
train_errors, val_errors = [], []

for epoch in range(1000):
    # Train and compute val. error:
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    # Save the best model:
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)   
    # Stop after n_iter_wait loops with no better val. error:
    if best_epoch < epoch - n_iter_wait:
        break

    # Save for plotting purpose:
    val_errors.append(val_error)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict)) 
train_errors = np.sqrt(train_errors) # convert to RMSE
val_errors = np.sqrt(val_errors)
# Print best epoch and model
best_epoch
best_model.intercept_, best_model.coef_  

# 9.5. Plot learning curves
if let_plot:
    best_val_error = val_errors[best_epoch]
    plt.plot(val_errors, "b-", linewidth=2, label="Validation set")
    plt.plot(train_errors, "r-", linewidth=2, label="Training set")
    plt.annotate('Best model',xytext=(best_epoch, best_val_error+0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 xy=(best_epoch, best_val_error), ha="center", fontsize=font_size,  )      
    plt.xlim(0, epoch)
    plt.grid()
    plt.legend(loc="upper right", fontsize=font_size)
    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Root Mean Squared Error", fontsize=font_size)
    plt.title("Learning curves w.r.t. the training time")
    plt.show()


# In[10]: LOGISTIC REGRESSION

# 10.1. Info (>> see slide)
if 0:
    # Plot log() function
    x = np.linspace(0.1,3,100)
    y = np.log(x)
    plt.plot(x,y,'b-', linewidth=2)
    plt.plot([0, 3],[0, 0],'k-')
    plt.xlabel('p')
    plt.ylabel('log(p)')
    plt.xlim(0,3)
    plt.grid()
    plt.savefig('figures/log.png')
    plt.show()

    # Plot -log() function
    y = -np.log(x)
    plt.plot(x,y,'b-', linewidth=2)
    plt.plot([0, 3],[0, 0],'k-')
    plt.xlabel('p')
    plt.ylabel('-log(p)')
    plt.xlim(0,3)
    plt.grid()
    plt.savefig('figures/neg_log.png')
    plt.show()

    # Plot -log(1-x) function
    y = -np.log(1-x)
    plt.plot(x,y,'b-', linewidth=2)
    plt.plot([0, 3],[0, 0],'k-')
    plt.xlabel('p')
    plt.ylabel('-log(1-p)')
    plt.xlim(0,3)
    plt.grid()
    plt.savefig('figures/neg_log_1_x.png')
    plt.show()

    # Plot sigmoid function
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    plt.savefig('figures/logistic_function_plot')
    plt.show()

# 10.2. Load Iris dataset 
from sklearn import datasets
iris = datasets.load_iris()
print(iris.keys()) # data: features, target: label
print(iris.DESCR) # description of the data

# 10.3. Train a logistic regression model 
X = iris["data"][:, 3:] # Use only 1 feature: petal width (max: 2.5 cm)  
y = (iris["target"] == 2).astype(np.int)  # 2 classes: True if Iris virginica, else False
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X, y)

# 10.4. Plot the learned model  
if let_plot:
    # Plot samples
    plt.figure(figsize=(8, 5))
    plt.plot(X[y==1], y[y==1], "gs", label="Iris virginica")
    plt.plot(X[y==0], y[y==0], "ro", label="Not Iris virginica")  
    
    # Plot the learned model (hypothesis function)
    X_test = np.linspace(0, 3, 1000).reshape(-1, 1) 
    y_proba = log_reg.predict_proba(X_test)        
    plt.plot(X_test, y_proba[:, 1], "b-", linewidth=2, label="Hypothesis")
    
    # Plot DECISION BOUNDARY (>> see slide)
    x_at_y_0_5 = -log_reg.intercept_ / log_reg.coef_[0] 
    #x_at_y_0_5 = X_test[y_proba[:,1] >= 0.5][0]
    plt.plot([x_at_y_0_5, x_at_y_0_5], [-0.03, 1.03], "k-", linewidth=3, label="Decision boundary")
    plt.plot([0, 3], [0.5, 0.5], "k:", linewidth=2)            

    plt.xlabel("Petal width (cm)", fontsize=font_size)
    plt.ylabel("Probability", fontsize=font_size)
    plt.legend(loc="upper left", fontsize=font_size)
    plt.axis([0, 3, -0.03, 1.03])         
    plt.show()

# 10.5. How many samples does the model misclassify?

# 10.6. Train another model, using 2 features 
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int) # 2 classes: True if Iris virginica, else False
log_reg = LogisticRegression(solver="lbfgs", random_state=42) #, C=10**10: smaller C => stronger regularization 
log_reg.fit(X, y)

if let_plot:
    # Plot samples
    plt.figure(figsize=(8, 5))
    plt.plot(X[y==1, 0], X[y==1, 1], "gs") #, label="Iris virginica"
    plt.plot(X[y==0, 0], X[y==0, 1], "ro") #, label="Not Iris virginica"
        
    # Contour plot of the hypothesis function      
    x0, x1 = np.meshgrid(
                np.linspace(2.9, 7, 500).reshape(-1, 1),
                np.linspace(0.8, 2.7, 200).reshape(-1, 1) )
    X_test = np.c_[x0.ravel(), x1.ravel()]   
    y_proba = log_reg.predict_proba(X_test)
    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, levels=30, cmap=plt.cm.RdYlGn) #brg
    plt.colorbar()
    #plt.clabel(contour, inline=1, fontsize=12)         

    # Plot decision boundary (>> see slide)
    x1 = np.array([2.9, 7])
    x2 = -(log_reg.coef_[0][0]*x1 + log_reg.intercept_) / log_reg.coef_[0][1]
    plt.plot(x1, x2, "k-", linewidth=3, label="Decision boundary")

    plt.text(3.5, 1.5, "Not Iris virginica", fontsize=font_size, color="r", ha="center")
    plt.text(6.5, 2.3, "Iris virginica", fontsize=font_size, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=font_size)
    plt.ylabel("Petal width", fontsize=font_size)
    plt.axis([2.9, 7, 0.8, 2.7])
    plt.legend(loc="upper left", fontsize=font_size)
    plt.show()


# In[11]: SOFTMAX REGRESSION

# 11.1. Info (>> see slide) 

# 11.2. Train a softmax model for Iris data
X = iris["data"][:, (2, 3)]   # petal length, petal width
y = iris["target"]  # use all 3 classes
softmax_reg = LogisticRegression(multi_class="multinomial", # multinomial: use Softmax regression
                                 solver="lbfgs", random_state=42) # C=10
softmax_reg.fit(X, y)

# 11.3. Try prediction
sample_id = 126 
softmax_reg.predict_proba([X[sample_id]]) 
softmax_reg.predict([X[sample_id]]) 
y[sample_id]

# 11.4. Plot hypothesis and decision boundary
if let_plot:
    # Plot samples:
    plt.figure(figsize=(10, 6))
    plt.plot(X[y==2, 0], X[y==2, 1], "bo", label="Iris virginica")
    plt.plot(X[y==1, 0], X[y==1, 1], "gs", label="Iris versicolor")
    plt.plot(X[y==0, 0], X[y==0, 1], "r*", label="Iris setosa")

    # Contour plot of hypothesis function of 1 class:
    x0, x1 = np.meshgrid(
                np.linspace(0, 8, 500).reshape(-1, 1),
                np.linspace(0, 3.5, 200).reshape(-1, 1) )
    X_test = np.c_[x0.ravel(), x1.ravel()]
    y_proba = softmax_reg.predict_proba(X_test)
    #z_hypothesis = y_proba[:, 0].reshape(x0.shape) # hypothesis of class 0: Iris setosa 
    z_hypothesis = y_proba[:, 1].reshape(x0.shape) # hypothesis of class 1: Iris versicolor
    #z_hypothesis = y_proba[:, 2].reshape(x0.shape) # hypothesis of class 2: Iris virginica
    contour = plt.contour(x0, x1, z_hypothesis, levels=25, cmap=plt.cm.Greens)
    #plt.clabel(contour, inline=1, fontsize=12)
    plt.colorbar()

    # Plot decision boundary (filled areas):
    y_predict = softmax_reg.predict(X_test)
    z_boundary = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#f7e1e1','#e1fae1','#c8dbfa'])
    plt.contourf(x0, x1, z_boundary, cmap=custom_cmap)
    
    plt.xlabel("Petal length", fontsize=font_size)
    plt.ylabel("Petal width", fontsize=font_size)
    plt.legend(loc="upper left", fontsize=font_size)
    plt.axis([0, 7, 0, 3.5])
    plt.title("Contour plot of hypothesis of class 1: Iris versicolor", fontsize=font_size)
    plt.show()


DONE = 1 




 


