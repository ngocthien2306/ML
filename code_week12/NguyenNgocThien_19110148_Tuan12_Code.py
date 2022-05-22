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
from sklearn.svm import LinearSVC 

'''
Thay đổi các phương trình tạo data giả lập, chọn các classes khác trong Iris dataset, và thực hiện chạy từng bước các thuật toán như demo trong bài học 
'''

# In[6]: SVM REGRESSION
# 6.1. Generata non-linear 1D data
np.random.seed(42)
m = 30 
X = 4*np.random.rand(m, 1) -2
y = (4 + 3*X**2 + X + np.random.randn(m, 1)).ravel()

# 6.2. Fit Linear SVM regressors
from sklearn.svm import LinearSVR
svm_reg1 = LinearSVR(epsilon=2.5, random_state=42)
svm_reg1.fit(X, y)
svm_reg2 = LinearSVR(epsilon=0.4, random_state=42)      
svm_reg2.fit(X, y)

# 6.3. Plot the hypothesis
#def find_support_vectors(svm_reg, X, y):
#    y_pred = svm_reg.predict(X)
#    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
#    return np.argwhere(off_margin)
def plot_svm_regression(svm_reg, X, y, axes):
    # Plot model, margins
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=3, label=r"Hypothesis $\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "b--", linewidth=1, label="Margins")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "b--", linewidth=1)
    
    # Mask violated samples:
    #plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    # Plot samples:
    plt.plot(X, y, "bo")
    
    plt.axis(axes)

let_plot=True
if let_plot:
    plt.figure(figsize=(9, 5))
    plt.subplot(1,2,1)
    xylim = [-2, 2, 3, 11]
    plot_svm_regression(svm_reg1, X, y, xylim)
    # Plot epsilon:
    x1_esp = 1
    y_esp = svm_reg1.predict([[x1_esp]])
    plt.plot([x1_esp, x1_esp], [y_esp, y_esp - svm_reg1.epsilon], "k-", linewidth=2)
    plt.annotate( '', xy=(x1_esp, y_esp), xycoords='data',
            xytext=(x1_esp, y_esp - svm_reg1.epsilon),
            textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 2.5}  )
    plt.text(x1_esp+.1, y_esp-svm_reg1.epsilon/2, r"$\epsilon$ = {}".format(svm_reg1.epsilon), fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.title(r"Model trained with $\epsilon$ = {}".format(svm_reg1.epsilon), fontsize=14)
    plt.ylabel(r"$y$", fontsize=14, rotation=0)
    
    plt.subplot(1,2,2)
    plot_svm_regression(svm_reg2, X, y, xylim) 
    plt.title(r"Model trained with $\epsilon = {}$".format(svm_reg2.epsilon), fontsize=14)
    plt.savefig("figs/05_SVM_reg_epsilon");
    plt.show()

#%%

def plot_samples(subplot, with_legend=False, with_ylabel=False):
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "b^", label="Iris virginica")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "go", label="Iris setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 7, 0, 3])
    if with_legend: plt.legend(loc="upper left", fontsize=14)
    if with_ylabel: plt.ylabel("Petal width", fontsize=14)

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
        
#%%
'''
2. Huấn luyện các classifiers LinearSVC, SVC, SGDClassifier trên một dataset tự tạo (dataset này có thể được phân lớp bằng linear boundary) và xem 3 classifiers này có thể tạo ra cùng một model không. Giải thích kết quả.

'''
import random
import numpy as np
'''
{'data': [fish]
 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3]),
 'target_names': array(['koi', 'arowana', 'shark', 'tilapia'], dtype='<U7'),
 'feature_names': array(['length (cm)', 'width (cm)', 'height (cm)'], dtype='<U11')}
'''
### dataset tự tạo (fish) thông số về 4 loại cá gồm chiều dài, rộng, cao. Có 40 sample cho mỗi loại
### 
def generate_fish_data():
    fish = []
    for i in range(0, 40):
        length = random.uniform(40, 60)
        width = random.uniform(20, 30)
        height = random.uniform(5, 30)
        fish.append([round(length, 2), round(width, 2), round(height,2)])


    for i in range(0, 40):
        length = random.uniform(4, 20)
        width = random.uniform(3,  9)
        height = random.uniform(5, 18)
        fish.append([round(length, 2), round(width, 2), round(height,2)])


    for i in range(0, 40):
        length = random.uniform(50, 60)
        width = random.uniform(10, 50)
        height = random.uniform(20, 80)
        fish.append([round(length, 2), round(width, 2), round(height,2)])

    for i in range(0, 40):

        length = random.uniform(10, 40)
        width = random.uniform(3, 8)
        height = random.uniform(6, 17)
        fish.append([round(length, 2), round(width, 2), round(height,2)])


    target = []
    for i in range(0, 160):
        if i < 40:
            target.append(0)
        elif i < 80 and i > 40:
            target.append(1)
        elif i < 120 and i > 80:
            target.append(2)
        else:
            target.append(3)  
                
    target_names = ['koi', 'arowana', 'shark', 'tilapia']
    feature_names = ['length (cm)', 'width (cm)', 'height (cm)']
    fish_data = {'data': np.array(fish), 'target': np.array(target), 'target_names': np.array(target_names), 'feature_names': np.array(feature_names)}
    return fish_data

#%%
# generate fish data
'''
Dataset tự tạo này nó phân lớp bằng binary boundary
'''
fish_data = generate_fish_data()
X = fish_data["data"][:, (0, 1)]  # length, width
y = fish_data["target"]
koi_or_arowana = (y == 0) | (y == 1)
X = X[koi_or_arowana]  
y = y[koi_or_arowana]



'''
Kết luận 3 classifiers này đều có dạng model bậc một, tuy nhiên đường thẳng mô tả dữ liệu khác nhau -> theta khác nhau (parameter)

Với LinearSVC ta có thể vẽ ra được dicision boundary với margin khá lớn
tuy nhiên support vector vẫn chưa tốt nhất vì nó có các sample nằm lệch ra

Với SVC ta có đường dicision boundary là tốt nhất support vector không để sample nào lọt qua

Với SGDCClassifier nó chỉ vẽ được trục cắt các dữ liệu ra, tuy nhiên không có đường support vector -> không có margin

'''

from sklearn.svm import LinearSVC # faster than SVC on large datasets
svm_clf = LinearSVC(C=1) # C: larger => 'harder margins'. loss = 'hinge': a loss of SVM
svm_clf.fit(X, y)
svm_clf.predict(X) # Predicted labels

plot_samples(subplot='133')
plot_svc_decision_boundary(svm_clf, 0, 100)
plt.title("Decision boundary of LinearSVC model", fontsize=14)
plt.savefig("figs/01_Decision_boundaries_LinearSVC_fish.png")
plt.show()



from sklearn.svm import SVC
svm_clf = SVC(kernel='linear', C=100)
svm_clf.fit(X, y)
svm_clf.predict(X) # Predicted labels

plot_samples(subplot='133')
plot_svc_decision_boundary(svm_clf, 0, 100)
plt.title("Decision boundary of SVM model", fontsize=14)
plt.savefig("figs/01_Decision_boundaries_SVC_fish.png")
plt.show()


from sklearn.linear_model import SGDClassifier
svm_clf = SGDClassifier(epsilon=1, random_state=42)
svm_clf.fit(X, y)
svm_clf.predict(X) # Predicted labels
plot_samples(subplot='133')
plot_svc_decision_boundary(svm_clf, 0, 100)
plt.title("Decision boundary of SVM model", fontsize=14)
plt.savefig("figs/01_Decision_boundaries_SGDClassifier_fish.png")
plt.show()




#%%
# 6.4. Which one fits the data better? (>> see slide)


#%% 6.5. Non-linear SVM regression
'''
3. Train polynomial SVM regression trong mục 6.5 sử dụng hyperparameters degree>2 và các giá trị khác nhau cho C và epsilon. Giải thích kết quả thu được.
'''
from sklearn.svm import SVR
# Recall: 
#   smaller epsilon ==> less data fitted (less overfitting)
#   smaller C ==> "softer" margins (less overfitting)
import numpy as np
np.random.seed(42)
m = 30 
X = 4*np.random.rand(m, 1) -2
y = (4 + 3*X**2 + X + np.random.randn(m, 1)).ravel()
'''
Điểm chung của 4 ví dụ: 
Kết quả thu được là một đường cong đi chéo qua các điểm dữ liệu, có chiều hướng giảm và có 1 đoạn bằng phẳng từ -1 đến 1,
'''
'''
Với degree = 5, epsilon = 0.3 và C=0.1 
Nó có margin nhỏ bởi vì epsilon nhỏ và các sample nằm trong margin cũng khá ít vì margin nhỏ và C nhỏ nó nên nó chấp 
nhận nhiều điểm nằm ngoài vì vậy đường regression nó chưa đi qua được nhiều điểm dữ liệu
'''
svm_poly_reg1 = SVR(kernel="poly", degree=5, epsilon=0.3, C=0.1, gamma="scale")
svm_poly_reg1.fit(X, y)

'''
Với degree = 5, epsilon = 0.3 và C=100
Nó có margin nhỏ bởi vì epsilon nhỏ và các sample nằm trong margin cũng khá ít vì margin nhỏ
Mặc dù có C lớn nó ít chấp nhận sample nằm ngoài, tuy nhiên với model bậc 5 thì đường cong của nó khó mô tả
được những dữ liệu bất thường này (các sample có giá trị khác biệt) -> sample nằm trong margin vẫn nhỏ -> chưa mô tả được dữ liệu
'''
svm_poly_reg2 = SVR(kernel="poly", degree=5, epsilon=0.3, C=100, gamma="scale")
svm_poly_reg2.fit(X, y)

'''
Với degree = 5, epsilon = 2 và C=0.1
Nó có margin lớn bởi vì epsilon lớn và các sample nằm trong margin cũng khá nhiều vì margin lớn
vì có C nhỏ nên có khá nhiều sample nằm ngoài đường desicion boundary. Nhưng khuynh hướng chung nó 
vẫn mô tả được hầu hết dữ liệu.
'''
svm_poly_reg3 = SVR(kernel="poly", degree=5, epsilon=2, C=0.1, gamma="scale")
svm_poly_reg3.fit(X, y)

'''
Với degree = 5, epsilon = 2 và C=100
Nó có margin lớn bởi vì epsilon lớn và các sample nằm trong margin cũng khá nhiều vì margin lớn
vì có C lớn nên có ít sample nằm ngoài đường desicion boundary hơn 3 ví dụ trên. Khuynh hướng chung nó 
vẫn mô tả được hầu hết dữ liệu -> tốt hơn các ví dụ trên
'''
svm_poly_reg4 = SVR(kernel="poly", degree=5, epsilon=2, C=100, gamma="scale")
svm_poly_reg4.fit(X, y)

'''
Kết luận chung: trong thực tế ta nên tránh trường hợp dữ liệu xấu như này (sample có giá trị chêch lệch), vì nó
ảnh hướng rất nhiều tới độ chính xác của model. Ngoài ra ta sử dụng các phương pháp làm giảm overfitting của model
như là regulazaiton, softmax,...
'''

if let_plot:
    plt.figure(figsize=(12, 9))
    plt.subplot(2,2,1)
    xylim = [-2, 2, 3, 11]
    plot_svm_regression(svm_poly_reg1, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg1.degree, svm_poly_reg1.epsilon, svm_poly_reg1.C), fontsize=14)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    
    plt.subplot(2,2,2)
    plot_svm_regression(svm_poly_reg2, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg2.degree, svm_poly_reg2.epsilon, svm_poly_reg2.C), fontsize=14)
    
    plt.subplot(2,2,3)
    plot_svm_regression(svm_poly_reg3, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg3.degree, svm_poly_reg3.epsilon, svm_poly_reg3.C), fontsize=14)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.xlabel(r"$x_1$", fontsize=14)
    
    plt.subplot(2,2,4)
    plot_svm_regression(svm_poly_reg4, X, y, xylim)
    plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg4.degree, svm_poly_reg4.epsilon, svm_poly_reg4.C), fontsize=14)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.show()

# 6.6. (exercise) Explain why epsilon=1 leads to wrong models (with both large and small C).
print("\n")
xylim = [-2, 2, 3, 11]
Gaus_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVR(kernel="rbf", gamma=5, C=0.001))  ])  
Gaus_kernel_svm_clf.fit(X, y)
Gaus_kernel_svm_clf.predict(X)

plt.subplot(1,2,1)
plot_svm_regression(Gaus_kernel_svm_clf, X, y, xylim)
plt.title(r"$degree={}, \epsilon={}, C={}$".format(svm_poly_reg4.degree, svm_poly_reg4.epsilon, svm_poly_reg4.C), fontsize=14)
plt.xlabel(r"$x_1$", fontsize=14)
plt.show()
#%%
def plot_predictions(clf, axes, no_of_points=500):
    x0 = np.linspace(axes[0], axes[1], no_of_points)
    x1 = np.linspace(axes[2], axes[3], no_of_points)
    x0, x1 = np.meshgrid(x0, x1)
    X = np.c_[x0.ravel(), x1.ravel()]

    # Plot predicted labels (decision boundary)
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.bwr, alpha=0.12)  
    
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "rs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bo")
    plt.axis(axes)
    #plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)
#%%

'''
Train một SVM regressor với data trong mục 6.5 sử dụng Gaussian RBF kernel với các giá trị khác nhau cho C và epsilon. Giải thích kết quả thu được.
'''
m = 30 
X1 = 4*np.random.rand(m, 1) -2
y1 = (4 + 3*X**2 + X + np.random.randn(m, 1)).ravel()
# Train 1 Gaussian SVM using Kernel trick 
Gaus_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVR(kernel="rbf", epsilon=0.6, C=0.001))  ])  
Gaus_kernel_svm_clf.fit(X1, y1)
Gaus_kernel_svm_clf.predict(X1)

# Train several Gaussian SVMs using Kernel trick 
epsilon1, epsilon2 = 0.2, 1, 3
C1, C2 = 0.01, 100, 10000
hyperparams = (epsilon1, C1), (epsilon1, C2), (epsilon2, C1), (epsilon2, C2)

svm_clfs = []
for epsilon, C in hyperparams:
    Gaus_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("", SVR(kernel="rbf", epsilon=epsilon, C=C)) ])
    Gaus_kernel_svm_clf.fit(X1, y1)
    svm_clfs.append(Gaus_kernel_svm_clf)

# Plot boundaries by different SVMs
plt.figure(figsize=(11, 9))
for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(2,2,i+1)
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X1, y1, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"Use Gaus. kernel with $\gamma = {}, C = {}$".format(gamma, C), fontsize=14)
    if i in (0, 1): 
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")
plt.show()


#%%
'''
5. Train SVM classifiers trên MNIST dataset (dùng one-versus-all). Thực hiện tuning các hyperparameters để đạt accuracy cao nhất có thể. 
'''
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 1.2. Reshape to 2D array: each row has 784 (28X28 pixel) features
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

svc_clf = SVC(kernel="linear", C=10000)
svr_clf = SVR(kernel="poly", degree=2, epsilon=0.7, C=10000)

from sklearn.multiclass import OneVsRestClassifier
ova_clf = OneVsRestClassifier(SVC(kernel="linear", C=10000, random_state=42))


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

param_grid = {'kernel':('linear', 'rbf'), 'C':[0.01, 1, 10, 100], 'gamma': 'scale'}

grid_search = GridSearchCV(ova_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train[:1000], y_train[:1000])
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)

y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))

# In[7]: SVM OUTLIER DETECTION 
# NOTE: 
#   OneClassSVM (UNsupervised fashion):  NOT effective
#   SVM classifier (supervised fashion: class normal and class outlier): BETTER

# 7.1. Generate non-linear 1D data
np.random.seed(42)
m = 50 
x1_normal = 4*np.random.rand(m, 1) -2
x1_outlier = 0.5*np.random.rand(3, 1)
x1 = np.c_[x1_normal.T,x1_outlier.T].T
x2_normal = 4 + 3*x1_normal**2 + x1_normal + np.random.randn(m, 1)
x2_outlier = 2*np.random.rand(3, 1) +12
x2 = np.c_[x2_normal.T,x2_outlier.T].T
X = np.c_[x1,x2]
X_normal = np.c_[x1_normal,x2_normal]
y = np.c_[np.ones(x1_normal.shape).T, np.zeros(x1_outlier.shape).T].ravel()

# [NOT GOOD] 7.2.1 Train a SVM outlier detector (unsupervised fashion)
# NOTE: only use NORMAL data in training
from sklearn.svm import OneClassSVM 
#ocsvm = OneClassSVM(kernel="rbf", nu=0.001, gamma=0.1)
ocsvm = OneClassSVM(kernel='poly', degree=2,coef0=10, gamma=0.1)
ocsvm.fit(X_normal)                       

# [BETTER] 7.2.2 Train a SVM classifier (supervised fashion)
from sklearn.svm import SVC 
svc = SVC(kernel="rbf", gamma=5, C=1)
svc.fit(X,y)

# 7.3. Plot outliers
if let_plot:     
    # Plot predicted normal samples
    y_pred = ocsvm.predict(X)
    #y_pred = svc.predict(X)
    id_normal = (y_pred==1)
    plt.plot(x1[id_normal], x2[id_normal], "bo", label=r"Predicted normal samples")
    
    # Plot outliers
    id_outlier = (y_pred==-1) # -1: outlier class in ocsvm
    #id_outlier = (y_pred==0) # 0: outlier class in SVC
    plt.plot(x1[id_outlier], x2[id_outlier], "ro", label=r"Predicted outliers")      

    # Plot decision boundary
    x1_mes, x2_mes = np.meshgrid(np.linspace(min(x1), max(x1), 50), np.linspace(min(x2), max(x2), 50));
    #scores = ocsvm.decision_function(np.c_[x1_mes.ravel(), x2_mes.ravel()])
    scores = svc.decision_function(np.c_[x1_mes.ravel(), x2_mes.ravel()])
    scores = scores.reshape(x1_mes.shape)
    plt.contour(x1_mes, x2_mes, scores, levels=[0], colors="blue")

    plt.legend(fontsize=12)
    plt.show()


# In[8]: MATH BEHIND SVM
# 8.1. Load data
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]  # use only 2 classes: setosa, versicolor
y = y[setosa_or_versicolor]

# 8.2. Fit a linear SVM
svm_clf = LinearSVC(C=1000) # larger C: less regularization
svm_clf.fit(X,y);

# 8.3. Plot data and decision function surface
def plot_3D_decision_function(w, b, x1_lim, x2_lim ):
    # require: pip install pyqt5
    #matplotlib qt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot samples
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], 0, "b^")
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], 0, "go")

    # Plot surface z=0
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    ax.plot_surface(x1, x2, np.zeros(x1.shape),  color="w", alpha=0.3) #, cstride=100, rstride=100)
                                                       
    # Plot decision boundary (and margins)
    m = 1 / np.linalg.norm(w)
    x2s_boundary = -x1s*(w[0]/w[1])-b/w[1]
    ax.plot(x1s, x2s_boundary, 0, "k-", linewidth=3, label=r"Decision boundary")
    x2s_margin_1 = -x1s*(w[0]/w[1])-(b-1)/w[1]
    x2s_margin_2 = -x1s*(w[0]/w[1])-(b+1)/w[1]         
    ax.plot(x1s, x2s_margin_1, 0, "k--", linewidth=1, label=r"Margins at h=1 and -1") 
    ax.plot(x1s, x2s_margin_2, 0, "k--", linewidth=1)
     
    # Plot decision function surface
    xs = np.c_[x1.ravel(), x2.ravel()]
    dec_func = (xs .dot(w) + b).reshape(x1.shape)      
    #ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    ax.plot_surface(x1, x2, dec_func, alpha=0.3, color="r")
    ax.text(4, 1, 3, "Decision function $h$", fontsize=12)       

    ax.axis(x1_lim + x2_lim)
    ax.set_xlabel(r"Petal length", fontsize=12, labelpad=10)
    ax.set_ylabel(r"Petal width", fontsize=12, labelpad=10)
    ax.set_zlabel(r"$h$", fontsize=14, labelpad=5)
    ax.legend(loc="upper left", fontsize=12)    
w=svm_clf.coef_[0]
b=svm_clf.intercept_[0]
plot_3D_decision_function(w,b,x1_lim=[0, 5.5],x2_lim=[0, 2])
plt.show()


#%% 8.4. Slope and margin (>> see slide)
def plot_2D_decision_function(w, b, x1_lim=[-3, 3]):
    # Plot decision function 
    x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
    y = w * x1 + b
    plt.plot(x1, y, linewidth=3, color="red", label="Decision func. h")

    # Plot margins at h=1 and h=-1
    m = 1 / w
    plt.plot([-m, m], [0, 0], "ko", linewidth=3, label="Margins at h=1 & -1")
    plt.plot([m, m], [0, 1], "k--", linewidth=1)
    plt.plot([-m, -m], [0, -1], "k--", linewidth=1)
    
    plt.axis(x1_lim + [-2, 2])
    plt.xlabel(r"$x_1$", fontsize=14)
    #plt.grid()
    plt.axhline(y=0, color='k')
    #plt.axvline(x=0, color='k')
    plt.title(r"Decision func. with $w_1 = {}$".format(w), fontsize=14)

plt.figure(figsize=(9, 5))
plt.subplot(1,2,1)
plot_2D_decision_function(1, 0)
plt.ylabel(r"h = $w_1 x_1$ + 0", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(1,2,2)
plot_2D_decision_function(0.5, 0) 
plt.show()
 
DONE = True



