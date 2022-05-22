'''
The following code is mainly from Chap 6, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb

LAST REVIEW: May 2022
'''

# In[0]: IMPORTS, SETTINGS
import sys
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20"  # Scikit-Learn ≥0.20 is required
import numpy as np
import os
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)  
let_plot = True

# In[1]: FIRST TREE 
# 1.1. Load data
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target # ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

# 1.2. Fit a decision tree
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)
# Try predicting class of a sample:
a_sample = [5, 2.5]
tree_clf.predict([a_sample])
tree_clf.predict_proba([a_sample]) # class probabilities

# 1.3. Plot the tree (acts as a hypothesis function)
from sklearn.tree import export_graphviz
export_graphviz( tree_clf,
        out_file=r"figs/iris_tree.dot",
        feature_names=["PETAL LENGTH","PETAL WIDTH"],
        class_names=['SETOSA', 'VERSICOLOR', 'VIRGINICA'],
        rounded=True, filled=True, leaves_parallel=True, 
        node_ids=True, proportion=False, precision=2 )

from graphviz import Source
# NOTE: Got ERRORs in the following code? To fix:
#   1. After pip install graphviz, you MUST download Graphviz and extract/install it to a folder (e.g., "D:\graphviz-2.44.1-win32")
#   2. Add evironment PATH for "Graphviz\bin\" folder (e.g., "D:\graphviz-2.44.1-win32\Graphviz\bin\")
Source.from_file(r"figs/iris_tree.dot").render(r"figs/iris_tree", format='pdf', view=True, cleanup=True)

#%% 1.4. Plot decision boundaries (>> see slide)
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[1, 7.5, 0, 2.5], iris=True, legend=False, plot_training=True):
    # Plot samples:
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "ro", label="Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "gs", label="Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "b^", label="Virginica")
        plt.axis(axes)
    
    # Plot decision boundaries:
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#a0faa0','#9898ff'])   # 'r','g','b'
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)
if let_plot:
    plt.figure(figsize=(8, 5))
    plot_decision_boundary(tree_clf, X, y, legend=True)
    plt.title("Decision boundaries by the tree")
    plt.savefig("figs/decision_tree_decision_boundaries.png")
    plt.show()


# In[2]: REGULARIZATION FOR DECISION TREE (>> see slide) 
# 2.1. Generate non-linear data
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=100, noise=0.3, random_state=53)

# 2.2. Train trees without and with regularization
deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=6, random_state=42)
#deep_tree_clf2 = DecisionTreeClassifier(max_depth=4, random_state=42)
deep_tree_clf2.fit(Xm, ym)
deep_tree_clf3 = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)
deep_tree_clf3.fit(Xm, ym)
deep_tree_clf4 = DecisionTreeClassifier(min_samples_leaf=15, random_state=42)
deep_tree_clf4.fit(Xm, ym)

'''
Kết quả thu được cho thấy khi ta cho min_samples_leaf càng cao thì đường decision boundary
càng không chính xác, có nhiều sample nằm sai vùng. Đường decision boundary càng đơn giản 
và ít bị phân chia hơn vì model bị đơn giản có thể dẫn đến underfitting.

Nguyên nhân có thể do dữ liệu nằm ở 1 node quá nhiều -> ít node -> model đơn giản. Cho phép min_samples_leaf càng lớn
đồng nghĩa các node sẽ chứa nhiều sample -> tăng cao khả năng chứa sample của class khác -> Gini impurity lớn 

Từ đó ta có thể thấy rằng min_samples_leaf càng nhỏ -> phân chia dữ liệu càng tốt. Có nhiều đường phân chia (gập gềnh nhưng không phải dạng đường cong)
Vì nó được sẽ ra từ giá trị của threshold ở các node. 
'''
# 2.3. Plot decision boundaries of trees
if let_plot:
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
    plt.title("No restrictions", fontsize=14)
    plt.subplot(122)
    plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
    plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
    plt.savefig("figs/min_samples_leaf_plot")
    plt.show()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_decision_boundary(deep_tree_clf3, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
    plt.title("min_samples_leaf = {}".format(deep_tree_clf3.min_samples_leaf), fontsize=14)
    plt.savefig("figs/min_samples_leaf_plot")
    plt.subplot(122)
    plot_decision_boundary(deep_tree_clf4, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
    plt.title("min_samples_leaf = {}".format(deep_tree_clf4.min_samples_leaf), fontsize=14)
    plt.savefig("figs/min_samples_leaf_plot")
    plt.show()





# In[3]: REGRESSION WITH DECISION TREES
# 3.1. Generate non-linear data
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 5*(X - 0.4)**2
y = y + np.random.randn(m, 1)/10
# Plot samples: 
if let_plot:
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)
    plt.grid()
    plt.axis([0, 1, -0.2, 1])
    #plt.savefig("figs/non_linear_data.png")
    plt.show()

# 3.2. Train decision tree regressors
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_reg1.fit(X, y)
tree_reg2 = DecisionTreeRegressor(max_depth=7, random_state=42)
tree_reg2.fit(X, y)
tree_reg3 = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_reg3.fit(X, y)
'''
Với parameter max_depth, độ lớn của cây sẽ tăng theo cấp số nhân và giá trị max_depth càng lớn
thì model càng dễ bị overfitting nên ta cần điều chỉnh sao cho phù hợp dựa vào số lượng sample để có thể tối ưu được. 
độ chính xác. Tuy nhiên có thể thấy rỏ ràng nếu độ sâu lớn thì số lượng sample ở mỗi nút càng nhỏ
và độ chính xác khá cao. MSE dường như gần bằng 0
'''
# 3.3. Plot decision trees
export_graphviz(tree_reg1,
        out_file=r"figs/regression_tree1.dot",
        feature_names=["x1"],
        rounded=True, filled=True,
        node_ids=True, precision=3)
Source.from_file(r"figs/regression_tree1.dot").render(r"figs/regression_tree1", format='pdf', view=True, cleanup=True)

export_graphviz(tree_reg2,
        out_file=r"figs/regression_tree2.dot",
        feature_names=["x1"],
        rounded=True, filled=True,
        node_ids=True, precision=3)
Source.from_file(r"figs/regression_tree2.dot").render(r"figs/regression_tree2", format='pdf', view=True, cleanup=True)

export_graphviz(tree_reg3,
        out_file=r"figs/regression_tree3.dot",
        feature_names=["x1"],
        rounded=True, filled=True,
        node_ids=True, precision=3)
Source.from_file(r"figs/regression_tree3.dot").render(r"figs/regression_tree3", format='pdf', view=True, cleanup=True)
# (>> see slide)

#%% 3.4. Plot regression hypotheses
def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    # Plot samples: 
    plt.plot(X, y, "b.")

    # Plot hypotheses: 
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.plot(x1, y_pred, "r-", linewidth=2, label=r"$\hat{y}$")
    
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel(ylabel, fontsize=14, rotation=0)
    
if let_plot:
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_regression_predictions(tree_reg1, X, y)
    #for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    #    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    #plt.text(0.21, 0.65, "Depth=0", fontsize=15)
    #plt.text(0.01, 0.2, "Depth=1", fontsize=13)
    #plt.text(0.65, 0.8, "Depth=1", fontsize=13)
    plt.legend(loc="upper center", fontsize=14)
    plt.title("Hypothesis trained with max_depth=2", fontsize=14)

    plt.subplot(122)
    plot_regression_predictions(tree_reg2, X, y, ylabel=None)
    #for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    #    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    #for split in (0.0458, 0.1298, 0.2873, 0.9040):
    #    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
    #plt.text(0.3, 0.5, "Depth=2", fontsize=13)
    plt.title("Hypothesis trained with max_depth=3", fontsize=14)

    plt.savefig("figs/tree_regression_plot")
    plt.show()

# (>> see slide)

#%% 3.5. Regularization
# Train models
tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg1.fit(X, y)

tree_reg2 = DecisionTreeRegressor(min_samples_leaf=3,random_state=42)
tree_reg2.fit(X, y)

tree_reg3 = DecisionTreeRegressor(min_samples_leaf=6,random_state=42)
tree_reg3.fit(X, y)

tree_reg4 = DecisionTreeRegressor(min_samples_leaf=16,random_state=42)
tree_reg4.fit(X, y)

'''
Giống với mục 2.2 khi ta tăng min_samples_leaf thì model sẽ càng đơn giản -> underfitting
Vì vậy với số liệu này ta có thể thấy predict càng không mô tả được dữ liệu 
khi min_samples_leaf càng lớn
'''
# Plot hypotheses
if let_plot:
    x1 = np.linspace(0, 1, 500).reshape(-1, 1)
    y_pred1 = tree_reg1.predict(x1)
    y_pred2 = tree_reg2.predict(x1)
    y_pred3 = tree_reg3.predict(x1)
    y_pred4 = tree_reg4.predict(x1)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y$", fontsize=14, rotation=0)
    plt.legend(loc="upper center", fontsize=14)
    plt.title("No restrictions", fontsize=14)
    plt.subplot(122)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=14)
    plt.title("Hypothesis with min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)
    plt.savefig("figs/tree_regression_regularization_plot1")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred3, "r.-", linewidth=2, label=r"$\hat{y}$")
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=14)
    plt.title("Hypothesis with min_samples_leaf={}".format(tree_reg3.min_samples_leaf), fontsize=14)
    plt.subplot(122)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred4, "r.-", linewidth=2, label=r"$\hat{y}$")
    plt.axis([0, 1, -0.2, 1.1])
    plt.xlabel("$x_1$", fontsize=14)
    plt.title("Hypothesis with min_samples_leaf={}".format(tree_reg4.min_samples_leaf), fontsize=14)
    plt.savefig("figs/tree_regression_regularization_plot2")
    plt.show()




# In[4]: LIMITATIONS OF DECISION TREES (>> see slide)
# 4.1. Sensitive to feature rotation
# Create data
np.random.seed(6)
m = 100
Xs = np.random.rand(m, 2) - 0.5
ys = (Xs[:, 0] > 0).astype(np.float32) * 2
# Rotate data
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xsr = Xs .dot (rotation_matrix)

# Train models with original and rotated data
tree_clf_s = DecisionTreeClassifier(min_samples_leaf=40, random_state=42) #min_samples_leaf=1
tree_clf_s.fit(Xs, ys)
tree_clf_sr1 = DecisionTreeClassifier(min_samples_leaf=15, random_state=42)
tree_clf_sr1.fit(Xsr, ys)
tree_clf_sr2 = DecisionTreeClassifier(min_samples_leaf=2, random_state=42)
tree_clf_sr2.fit(Xsr, ys)
'''
1.  
min_samples_leaf -> đường decision boundary càng phức tạp. Tuy nhiên nó mô tả được hầu như các dữ liệu
'''

'''
2. Giải thích tại sao phải dùng min_samples_leaf nhỏ mới classify được rotated data trong mục 4.1. 
Trả lời: Bởi vì min_samples_leaf càng nhỏ đường decision boundary càng phức tạp, mới có tả mô tả được dữ liệu rotated
Nếu như min_samples_leaf thì đường decision boundary dường như là đường thẳng -> không mô tả tốt dữ liệu rotated

Đặc thù của đường decision boundary trong decision tree và nó vẽ dựa trên dữ trên chọn feature và giá trị threshold.
Nó có dạng đường thẳng đi tới các điểm (giá trị của threshold). Gập lên, xuống trái phải nhưng không phải dạng đường cong
'''
# Plot decision boundaries
if let_plot:
    plt.figure(figsize=(17, 5))
    plt.subplot(131)
    plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
    plt.title("Original data. \nModel with min_samples_leaf={}".format(tree_clf_s.min_samples_leaf))
    plt.subplot(132)
    plot_decision_boundary(tree_clf_sr1, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
    plt.title("Rotated data. \nModel with min_samples_leaf={}".format(tree_clf_sr1.min_samples_leaf))
    plt.subplot(133)
    plot_decision_boundary(tree_clf_sr2, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
    plt.title("Rotated data. \nModel with min_samples_leaf={}".format(tree_clf_sr2.min_samples_leaf))
    plt.savefig("figs/sensitivity_to_rotation_plot")
    plt.show()

#%%
'''
3.Huấn luyện và tunning một decision tree cho data tạo bằng lệnh: make_moons(n_samples=10000, noise=0.4).
    a. Chia data thành training set và test set sử dụng train_test_split().
    b. Dùng grid search với cross-validation (GridSearchCV) để tìm giá trị hyperparameters cho DecisionTreeClassifier. 
    Gợi ý: thử các giá trị max_leaf_nodes khác nhau. 
    c. Tính accuracy của model tốt nhất. Accuracy phải đạt >85%.
'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

'''
{'max_depth': 4, 'max_leaf_nodes': 16, 'min_samples_leaf': 12}
Accuracy = 0.852 -> 85.2%
'''
X, y = make_moons(n_samples=10000, noise=0.4, random_state=53)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dst = DecisionTreeClassifier()
params_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15], 
               'min_samples_leaf': [4, 5, 6, 7, 9, 12, 15, 20],
               'max_leaf_nodes': [2, 4, 8, 10, 12, 16, 24, 32, 64, 128, 512, 4096]}
grid_search = GridSearchCV(dst, params_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)

y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))
#%% 4.2. Output models changed from one train to another
# Load Iris data
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target # ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

# Train decision trees 2 times (with different random seeds)
tree_clf1 = DecisionTreeClassifier(random_state=40)
tree_clf1.fit(X, y)
tree_clf2 = DecisionTreeClassifier(random_state=41)
tree_clf2.fit(X, y)

# Plot decision trees
if let_plot:
    plt.figure(figsize=[12,5])
    plt.subplot(121)
    plot_decision_boundary(tree_clf1, X, y, legend=False)
    plt.title("First train")
    plt.subplot(122)
    plot_decision_boundary(tree_clf2, X, y, legend=False)
    plt.title("Second train")
    plt.savefig("figs/decision_tree_instability_plot")
    plt.show()


DONE = True


