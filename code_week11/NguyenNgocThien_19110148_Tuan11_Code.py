'''
The following code is mainly from Chap 5, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/05_support_vector_machines.ipynb

LAST REVIEW: April 2022
'''
# Source code của thầy

# In[0]: IMPORTS, SETTINGS
import sklearn 
assert sklearn.__version__ >= "0.20" # sklearn ≥0.2 is required
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(42) # to output the same across runs
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[1]: LINEAR DECISION BOUNDARIES      
# 1.1. Load Iris dataset
'''

'''
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # sepal length, sepal width, petal length, petal width
y = iris["target"]
'''
Sử dụng 2 class khác là setosa và virgincica
'''
setosa_or_virginica = (y == 0) | (y == 2)
X = X[setosa_or_virginica]  # use only 2 classes: setosa, virginica
y = y[setosa_or_virginica]

def plot_samples(subplot, with_legend=False, with_ylabel=False):
    plt.subplot(subplot)
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "b^", label="Iris virginica")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "go", label="Iris setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    if with_legend: plt.legend(loc="upper left", fontsize=14)
    if with_ylabel: plt.ylabel("Petal width", fontsize=14)

# 1.2. Decision boundaries of arbitrary models
'''
Tạo model cho Decision boundaries
'''
x1 = np.array([0, 5.5]) # points to plot
x2_model_1 = 4*x1 - 18
x2_model_2 = 1.2*x1 - 2.5

# 1.3. Train a linear SVM classifier model
# 3 implementation of linear SVM classifiers: 
# 1.
# from sklearn.svm import SVC 
# svm_clf = SVC(kernel="linear", C=np.inf):  it's SLOW
# 2.
# from sklearn.linear_model import SGDClassifier
# SGDClassifier(loss="hinge", alpha=1/(m*C)): not as fast as LinearSVC(), but works with huge datasets   
# 3.
from sklearn.svm import LinearSVC # faster than SVC on large datasets
svm_clf = LinearSVC(C=np.inf) # C: larger => 'harder margins'. loss = 'hinge': a loss of SVM
svm_clf.fit(X, y)
svm_clf.predict(X) # Predicted labels

# 1.4. Plot decision boundaries of models
# Plot arbitrary model 1:
'''
Đường Decision boundary chưa chia được các loại dữ liệu ra riêng, có vài sample của class Virginica 
nằm qua bên Setosa -> khi thêm sample vào -> có sai số, kém hiệu quả
'''
plt.figure(figsize = [16, 5])
plot_samples(subplot='131', with_legend=True, with_ylabel=True)
plt.plot(x1, x2_model_1, "k-", linewidth=3)
plt.title("Decision boundary of model 1", fontsize=14)
'''
Đường Decision boundary đã chia được dữ liệu ra làm hai phần riêng biệt
tuy nhiên, đường thằng chia cắt chưa hoàn hảo khi ta thêm vài sample 
có thể nó sẽ nằm sai vùng. Vì khoảng cách các sample nằm gần đường thẳng chia cách khá là gần
'''
# Plot arbitrary model 2:
plot_samples(subplot='132')
plt.plot(x1, x2_model_2, "k-", linewidth=3)
plt.title("Decision boundary of model 2", fontsize=14)

# Plot SVM model:
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # Plot decision boundary:
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]        
    x1 = np.linspace(xmin, xmax, 200)
    x2 = -w[0]/w[1]*x1 - b/w[1] # Note: At the decision boundary, w1*x1 + w2*x2 + b = 0 => x2 = -w1/w2 * x1 - b/w2
    plt.plot(x1, x2, "k-", linewidth=3, label="SVM")
    
    # Plot gutters of the margin:
    margin = 1/w[1]
    right_gutter = x2 + margin
    left_gutter = x2 - margin
    plt.plot(x1, right_gutter, "k:", linewidth=2)
    plt.plot(x1, left_gutter, "k:", linewidth=2)

    # Highlight samples at the gutters (support vectors):
    skipped=True
    if not skipped:
        hinge_labels = y*2 - 1 # hinge loss label: -1, 1. our label y: 0, 1
        scores = X.dot(w) + b
        support_vectors_id = (hinge_labels*scores < 1).ravel()
        svm_clf.support_vectors_ = X[support_vectors_id]      
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')

'''
Kết quả cho thấy đường thẳng phân chia các class rất là tốt, khoảng cách (margin)
từ sample của 2 class gần đường thẳng nhất khá là xa (xa hơn 2 đường trên)-> khi thêm sample sẽ cho kết quả tốt hơn
thậm chí là không có sample của class khác nhầm qua class nọ
'''
plot_samples(subplot='133')
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.title("Decision boundary of SVM model", fontsize=14)
#plt.savefig("figs/01_Decision boundaries.png")
plt.show()

# 1.5. Large vs small margins (>> see slide)
'''
Large margins là khoảng cách từ trục cho đến đường biên (chạm tới sample gần nhất) có khoảng cách lớn, 
ngược lại small margin có khoảng cách nhỏ
'''
# 1.6. Support vectors (>> see slide) 
'''
Support vectors và vector có samole nằm trên (chạm) vector đó
'''
if 0:
    # Plot SVM model in a separate figure
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "b^")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "go")
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis("square")
    plt.axis([0, 5.5, -1, 2.5])
    plt.title("Decision boundary of SVM model", fontsize=14)
    plt.savefig("figs/02_Linear_SVM")
    plt.show()


# [SKIP] 1.7 Sensitivity to feature scales
if 0:
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 2, 2])
    from sklearn.svm import SVC 
    svm_clf = SVC(kernel="linear", C=100)
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(9,2.7))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")
    plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")
    plot_svc_decision_boundary(svm_clf, 0, 6)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x_1$", fontsize=20, rotation=0)
    plt.title("Unscaled", fontsize=16)
    plt.axis("square")
    #plt.axis([0, 6, 0, 90])
    plt.axis([0, 6, 45, 65])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(122)
    plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")
    plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 1][ys==0], "ms")
    plot_svc_decision_boundary(svm_clf, -2, 2)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x'_1$  ", fontsize=20, rotation=0)
    plt.title("Scaled", fontsize=16)
    plt.axis("square")
    #plt.axis([0, 20, 0, 80])
    plt.axis([-2, 2, -2, 2])
    plt.show()


# In[2]: HARD MARGIN VS SOFT MARGIN

# 2.1. Hard margin (>> see slide)
'''
All sample thuộc 1 class -> phải bên 1 area
Vì vậy nếu có abnormal sample nó buộc tìm ra 1 đường thẳng nào đó thỏa mãn
Tuy nhiên trong thực tế có những dữ liệu rất bất thường dẫn đến margin sẽ cực nhỏ
Thậm chí là không có đường nào thỏa mãn điều kiện
'''
# 2.1.1. Problem 1: Only works with linearly separate data
# Add an abnormal sample 
Xo1 = np.concatenate([X, [[4.2, 1.6]]], axis=0)
yo1 = np.concatenate([y, [0]], axis=0)       
# Plot new training data
let_plot=True
if let_plot:
    plt.plot(Xo1[:, 0][yo1==2], Xo1[:, 1][yo1==2], "b^")
    plt.plot(Xo1[:, 0][yo1==0], Xo1[:, 1][yo1==0], "go")
    #plt.text(0.4, 1.8, "Impossible!", fontsize=16, color="red")
    plt.annotate("Outlier", xytext=(2.6, 1.5),
                 xy=(Xo1[-1][0], Xo1[-1][1]),
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 ha="center", fontsize=14 )
    plt.axis([0, 5.5, 0, 2])
    plt.title("Can a linear model fit this data?", color="red", fontsize=14)
    plt.show()

#%% 2.1.2. Problem 2: Sensitive to outliners 
# Add an abnormal sample 

Xo2 = np.concatenate([X, [[3.2, 0.8]]], axis=0)
yo2 = np.concatenate([y, [0]], axis=0)    
# Train and plot SVM models  
svm_clf2 = LinearSVC(C=np.Inf, max_iter=5000, random_state=42)
svm_clf2.fit(Xo2, yo2)
if let_plot:
    # Plot SVM trained without outlier 
    plt.figure(figsize = [10, 5])
    plt.subplot(1,2,1)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "b^")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "go")    
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.title("SVM trained without outliner", fontsize=14 )
    plt.axis([0, 5.5, 0, 2])

    # Plot SVM trained with outlier 
    plt.subplot(1,2,2)
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "b^")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "go")
    plt.annotate("Outlier", xytext=(3.2, 0.26),
                 xy=(Xo2[-1][0], Xo2[-1][1]),
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 ha="center", fontsize=14 )
    plot_svc_decision_boundary(svm_clf2, 0, 5.5)
    plt.title("SVM trained with outliner", fontsize=14 )
    plt.axis([0, 5.5, 0, 2])
    plt.show()                


#%% 2.2. Soft margin (>> see slide)
# 2.2.1. Fit SVM models
svm_clf1 = LinearSVC(C=3, random_state=42) #, loss="hinge": standard loss for classification
svm_clf1.fit(Xo2, yo2)   
svm_clf2 = LinearSVC(C=1000, random_state=42)
svm_clf2.fit(Xo2, yo2)
'''
Với parameter C càng lớn thì nó sẽ càng nghiêm ngặt (cứng), càng không chấp nhận các sample
nằm ngoài đường dicision boundary -> margin nhỏ
ngược lại C càng nhỏ -> mềm dẽo -> chấp nhận sample nằm ngoài.
Tuy nhiên trong thực tế nó vẫn tốt vì có margin lớn -> chỉ sai số nhỏ và không đáng để
'''
# 2.2.2. Plot decision boundaries and margins
if let_plot:
    plt.figure(figsize=[10, 5])
    plt.subplot(1,2,1)
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "b^", label="Iris virginica")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "go", label="Iris versicolor")
    plt.legend(loc="upper left", fontsize=12)
    plot_svc_decision_boundary(svm_clf1, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.title("LinearSVC with C = {}".format(svm_clf1.C), fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(1,2,2)
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "b^", label="Iris virginica")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "go", label="Iris versicolor")
    plot_svc_decision_boundary(svm_clf2, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.title("LinearSVC with C = {}".format(svm_clf2.C), fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    plt.savefig("figs/03_Different C values.png")
    plt.show()


# In[3]: NONLINEAR SVM 

# 3.1. Intro (>> see slide)

# 3.2. Load non-linear data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "rs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bo")
    plt.axis(axes)
    #plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)


'''
noise càng lớn dữ liệu càng nhiễu với noise = 0 -> dữ liệu giống hình moon nhất
if > 0.2 dữ liệu càng nhiễu đi 
'''  
for i in range(0, 11):
    X, y = make_moons(n_samples=100, noise=i/10, random_state=42)
    print(i)
    if let_plot:
        plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        plt.title("noise: " + str(i/10))
        #plt.savefig("figs/04_Nonlinear_data.png");
        plt.show()
        
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "rs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bo")
    plt.axis(axes)
    #plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)



# In[4]: METHOD 1 FOR NONLINEAR DATA: ADD POLINOMIAL FEATURES AND TRAIN LINEAR SVM
# 4.1. Add polinomial features and train linear svm 
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=40, random_state=42)) ])
polynomial_svm_clf.fit(X, y)

# Plot decision boundary
def plot_predictions(clf, axes, no_of_points=500):
    x0 = np.linspace(axes[0], axes[1], no_of_points)
    x1 = np.linspace(axes[2], axes[3], no_of_points)
    x0, x1 = np.meshgrid(x0, x1)
    X = np.c_[x0.ravel(), x1.ravel()]

    # Plot predicted labels (decision boundary)
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.bwr, alpha=0.12)  
    
    # Contour plot of samples' scores  
    #y_decision = clf.decision_function(X).reshape(x0.shape)
    #plt.contourf(x0, x1, y_decision, cmap=plt.cm.bwr, alpha=0.5)
    #plt.colorbar()
'''
poly_features degree càng cao -> đường phân tách dữ liệu càng ngoằn ngèo, dao động mạnh
Tuy nhiên vẫn chưa phân chia tốt được dữ liệu -> sample giống nhau lại nằm phía khác nhau
'''
if let_plot:
    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()

#%% 4.2. Kernel trick for method 1: Polynomial kernel
'''
Đễ giải quyết vấn đề trên (poly_features degree cao) ta sử dụng kernel trick
tạo ra desicion boundary phức tạp gần bằng bậc cao nhưng không cần thên feature bậc cao
'''
from sklearn.svm import SVC
# NOTE: 
#   larger coef0 => the more the model is influenced by high-degree polynomials
def poly_svm_model(degree, coef, c):
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=degree, coef0=coef, C=c))  ]) 
    return svm

def plot_by_svm_model(poly_svm):
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plot_predictions(poly_svm, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
    plt.title(r"degree={}, coef0={}, C={}".format(poly_svm[1].degree,poly_svm[1].coef0,poly_svm[1].C), fontsize=14)
    
'''
Dựa vào 8 model sau ta có thể thấy rằng bậc (degree) của model càng lớn -> chưa mô tả hiệu quả dữ liệu vì bậc càng cao 
thì dạng đường cong của nó càng phức tạp uống cong nhiều chỗ có thể thấy rõ hơn ở poly_svm_3, poly_svm_4
Với C nhỏ -> sample cùng 1 loại dữ liệu lại nằm nhiều hơn ở các vùng khác. Nó chấp nhận nhiều sample nằm sai
Ngược lại thì nó càng ít chấp nhận sample nằm sai vùng, ta có thể thấy rõ sự khác biệt ở  poly_svm_1, poly_svm_2

Với coef0 càng lớn thì độ uống cong của nó càng ít -> mô tả dữ liệu rất tốt kèm với Degree nhỏ và C lớn (nhưng trong thực tế sẽ rất khó khi dữ liệu bất thường)
Tuy nhiên sau khi quan sát các trường hợp thì em cho rằng Degree nên < 5, C > 10 < 50 và coef0 khoảng 0.5 - 1 là khá tốt
'''    
poly_svm_1 = poly_svm_model(5, 0.002, 1)
poly_svm_2 = poly_svm_model(5, 0.002, 300)
poly_svm_3 = poly_svm_model(10, 0.002, 1)
poly_svm_4 = poly_svm_model(10, 0.002, 300)
poly_svm_5 = poly_svm_model(5, 80, 1)
poly_svm_6 = poly_svm_model(5, 80, 300)
poly_svm_7 = poly_svm_model(10, 80, 1)
poly_svm_8 = poly_svm_model(10, 80, 300)
plot_by_svm_model(poly_svm_1, 1, 2, 1)
plot_by_svm_model(poly_svm_2, 1, 2, 2)
plot_by_svm_model(poly_svm_3, 1, 2, 1)
plot_by_svm_model(poly_svm_4, 1, 2, 2)
plot_by_svm_model(poly_svm_5, 1, 2, 1)
plot_by_svm_model(poly_svm_6, 1, 2, 2)
plot_by_svm_model(poly_svm_7, 1, 2, 1)
plot_by_svm_model(poly_svm_8, 1, 2, 2)
plt.show()



# In[5]: METHOD 2: ADD SIMILARITY FEATURES AND TRAIN 
# 5.1. Generate 1-fearture data (1-dimenstional data)
X_1D = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]).reshape(-1,1) 
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0]) # 2 classes

# 5.2. Plot Gaussian kernel graphs
def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)
def plot_kernel(X_1D,y,landmark,gamma, no_plot_points=200, xy_lim = [-4.5, 4.5, -0.1, 1.1]):  
    # Plot samples:
    plt.axhline(y=0, color='k') # Ox axis
    plt.plot(X_1D[y==0], np.zeros(4), "rs", markersize=9, label="Data samples (class 0)")
    plt.plot(X_1D[y==1], np.zeros(5), "g^", markersize=9, label="Data samples (class 1)")

    # Plot the landmark:
    plt.scatter(landmark, [0], s=200, alpha=0.5, c="orange")
    plt.annotate(r'landmark',xytext=(landmark, 0.2),
                 xy=(landmark, 0), ha="center", fontsize=14,
                 arrowprops=dict(facecolor='black', shrink=0.1)  )
    
    # Plot Gaussian kernel graph: 
    x1_plot = np.linspace(-4.5, 4.5, no_plot_points).reshape(-1,1)  
    x2_plot = gaussian_rbf(x1_plot, landmark, gamma)
    plt.plot(x1_plot, x2_plot, "b--", linewidth=2, label="Gaussian kernel")
    
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=13)
    #plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.axis(xy_lim)
    plt.title(r"Gaussian kernel with $\gamma={}$".format(gamma), fontsize=14)

# Gaussian kernel 1
landmark1 = np.array([-1.5])
gamma1 = 0.16
if let_plot:
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plt.legend(fontsize=12, loc="upper right")
    plt.show()

# Gaussian kernel 2: larger gamma, more concentrate around the landmark
landmark2 = np.array([0.26])
gamma2 = 0.51
if let_plot:
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=12, loc="upper right")
    plt.show()


#%% 5.3. Data transformation (>> see slide) 
def plot_transformed_data(X_2D,y,xy_lim=[-4.5, 4.5, -0.1, 1.1]):
    plt.axhline(y=0, color='k') # Ox
    #plt.axvline(x=0, color='k') # Oy
    plt.plot(X_2D[:, 0][y==0], X_2D[:, 1][y==0], "rs", markersize=9, label="Samples (class 0)")
    plt.plot(X_2D[:, 0][y==1], X_2D[:, 1][y==1], "g^", markersize=9, label="Samples (class 1)")

    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$ (similarity feature)", fontsize=14)
    plt.axis(xy_lim)
    plt.title("Data in new feature space", fontsize=14)

# 5.3.1. Use Gaussian kernel 1 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1)]
    plot_transformed_data(X_2D,y)
    plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()

# 5.3.2. Use Gaussian kernel 2 with 1 landmark (add 1 feature)
if let_plot:
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark2,gamma2)    
    #plt.legend(fontsize=10, loc="upper right")

    plt.subplot(122)
    X_2D = np.c_[X_1D, gaussian_rbf(X_1D, landmark2, gamma2)]
    plot_transformed_data(X_2D,y)
    #plt.legend(fontsize=12, loc="upper right")
    plt.ylabel("$x_2$",fontsize=14)
    plt.show()


# 5.3.3. Use Gaussian kernels with 2 landmarks (add 2 features)
if let_plot:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plot_kernel(X_1D,y,landmark1,gamma1)    
    plot_kernel(X_1D,y,landmark2,gamma2)    
    plt.title("2 Gaussian kernels", fontsize=14)#plt.legend(fontsize=10, loc="upper right")

    #from mpl_toolkits.mplot3d import Axes3D 
    ax = fig.add_subplot(122, projection='3d')
    X_3D = np.c_[X_1D, gaussian_rbf(X_1D, landmark1, gamma1), 
                 gaussian_rbf(X_1D, landmark2, gamma2)]
    ax.scatter(X_3D[:, 0][y==0], X_3D[:, 1][y==0], X_3D[:, 2][y==0], 
                s=115,c="red",marker='s',label="Samples (class 0)")
    ax.scatter(X_3D[:, 0][y==1], X_3D[:, 1][y==1], X_3D[:, 2][y==1], 
                s=115,c="green",marker='^',label="Samples (class 1)")

    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$\n(similarity to lm 1)", fontsize=12)
    ax.set_zlabel("$x_3$\n(similarity to lm 2)", fontsize=12)
    plt.title("Data in new feature space", fontsize=14)
    plt.show()


#%% 5.4. How to choose landmarks? (>> see slide)
'''
Với noise = 0.3, gamma = 0.5 và C = 0.01 ta có thể thấy dữ liệu rất nhiễu
và margin khá là 'mền' cho nhiều điểm đi lọt vì C nhỏ. Thêm 1 điểm đáng chú ý là
đường mô tả dữ liệu chưa phức tạp độ cong ít, không nhiều khúc khiểu vì có gamma nhỏ
--> underfiting

Với noise = 0.3 gamma = 0.5 và c = 500 điểu khác biệt là với c lớn -> cho nên các sample ít lọt qua margin hơn -> vì vậy đường cong nó dao động nhiều hơn
mặc dù cùng gamma. Dao động nhiều vì tính chất của hard margin là all sample thuộc 1 class -> 1 side
Vì vậy khi nó cố làm điều đó với model non-linear -> nó tạo độ cong mạnh
--> ổn định

Với noise = 0.3 gamma = 10 và c = 0.01 điểm dữ liệu ít bị nhiễu có dạng đường cong 
và margin khá là 'mền' cho nhiều điểu đi lọt vì C nhỏ giống với hình 1
Tuy nhiên với gamma lớn nó đã tạo ra dạng đường mô tả dữ liệu khá là khúc khiểu
--> overfiting

Với noise = 0.3 gamma = 10 và c = 500 đường cong mạnh, nhiều dao động và khúc khiểu hơn hình 3
Vì tính cứng của margin khi c lớn
--> overfiting nặng nhất

Khi noise = 0.1 và các thông số gamma và c tương tự trên thì nó mô tả dữ liệu cực kì tốt
ngoại trừ hình 1 vì nó bị underfiting
hình 2 là tốt nhất

- noise = 0.1, gamma = 0.5 và C = 0.01 --> underfiting khá nặng
- noise = 0.1, gamma = 0.5 và C = 500 --> ổn định -> tốt nhất
- noise = 0.1, gamma = 10 và C = 0.01 --> overfiting nặng nhất 
- noise = 0.1, gamma = 10 và C = 500 --> overfitting

Model tốt hay không cũng phục thuộc vào dữ liệu, tuy nhiên trong trường hợp dữ liệu xấu
ta có thể chọn gamma nhỏ và C vừa phải thì model sẽ được tối ưu nhất trong các ví dụ ở trên
'''

# 5.6. Kernel trick for method 2 (Gaussian kernel)
# Generate non-linear data
X, y = make_moons(n_samples=100, noise=0.3, random_state=42)
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
# Train 1 Gaussian SVM using Kernel trick 
Gaus_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))  ])  
Gaus_kernel_svm_clf.fit(X, y)
Gaus_kernel_svm_clf.predict(X)

# Train several Gaussian SVMs using Kernel trick 
gamma1, gamma2 = 0.5, 10
C1, C2 = 0.01, 500
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    Gaus_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("", SVC(kernel="rbf", gamma=gamma, C=C)) ])
    Gaus_kernel_svm_clf.fit(X, y)
    svm_clfs.append(Gaus_kernel_svm_clf)

# Plot boundaries by different SVMs
plt.figure(figsize=(11, 9))
for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(2,2,i+1)
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"Use Gaus. kernel with $\gamma = {}, C = {}$".format(gamma, C), fontsize=14)
    if i in (0, 1): 
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")
plt.show()

# 5.7. (>> see slide) What is the effect of: 
#   Large / small C?
#   Large / small gamma: ?

