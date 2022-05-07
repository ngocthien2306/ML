'''
The following code is mainly from Chap 3, Géron 2019 
https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb

LAST REVIEW: March 2022
'''

'''
Source code của Thầy
'''

"""
Thử Random Forest Classifier (sklearn.ensemble.RandomForestClassifier) và so sánh kết quả với SGDClassifier (linear model). 
Em thử 2 model RandomForestClassifier và SGDClassifier dùng file Jupyter, vì điều này sẽ dễ dàng theo dõi kết quả hơn ở mỗi bước chạy, 
cũng như compare kết quả giữa 2 model
"""
# In[0]: IMPORTS
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import joblib # Note: require sklearn v0.22+ (to update sklearn: pip install -U scikit-learn ). For old version sklearn: from sklearn.externals import joblib 
from sklearn.linear_model import SGDClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import f1_score

# In[1]: fashion_mnist DATASET
# 1.1. Load fashion_mnist     
"""
Description dataset: Đây là 1 tập dữ liệu về thời trang, có khoảng 70000 sameple bao gồm 10 loại phụ kiện thời trang khác nhau,
được sắp xếp ngẫu nhiên và biểu diễn hình ảnh dưới dạng 28x28 pixel và trên nền màu trắng đen (Gray)
Chi tiết các loại phụ kiện như sao
label 0: áo thun tay ngắn, cổ tròn
label 1: thuộc nhóm quần jean
label 2: áo thun mùa đông (sweater), ôm cổ, tay dài
label 3: liên quan đến nhóm phụ kiện váy dài ôm body
label 4: là nhóm phụ kiện thuộc áo khoác dài tay
label 5: là các phụ kiện thời trang thuộc loại giày sandal
label 6: nhóm phụ kiện này liên quan tới áo thun
label 7: phụ kiện thời trang thuộc loại giày adidas 
label 8: các phụ kiện liên quan đến túi sách
label 9: giày cao cổ (boot)
""" 

# dùng thư viện tensoflow để lấy dữ liệu
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 1.2. Reshape to 2D array: each row has 784 (28X28 pixel) features
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

# 1.3. Plot a digit image   
import random 
def plot_digit(data, label = 'unspecified', showed=True):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("Digit: " + str(label))
    #plt.axis("off")
    if showed:
        plt.show()
for i in range(10): 
    sample_id = random.randint(1, 1000)
    plot_digit(X_train[sample_id], y_train[sample_id])


# In[2]: TRAINING A BINARY CLASSIFIER (just two classes, 5 and not-5)               
# 2.1. Create label array: True for label 2 (áo thun mùa đông), False for other digits.
y_train_2 = (y_train == 2) 
y_test_2 = (y_test == 2)

# 2.2. Try Stochastic Gradient Descent (SGD) classifier
# Note 1: In sklearn, SGDClassifier train linear classifiers using SGD, 
#         depending on the loss, eg. ‘hinge’ (default): linear SVM, ‘log’: logistic regression, etc.
# Note 2: SGD takes 1 datum at a time, hence well suited for online learning. 
#         It also able to handle very large datasets efficiently
from sklearn.linear_model import SGDClassifier       
sgd_clf = SGDClassifier(random_state=42) # set random_state to reproduce the result
# Train: 
# Warning: takes time for new run!
new_run = True
if new_run == True:
    # hàm fit dùng để tính toán và training model 
    sgd_clf.fit(X_train, y_train_2)
    joblib.dump(sgd_clf,'saved_var/sgd_clf_binary')
else:
    sgd_clf = joblib.load('saved_var/sgd_clf_binary')


randomForest = True
rfc = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=1, random_state=101, max_features=None, min_samples_leaf=3)
if randomForest:
  rfc.fit(X_train, y_train_2)
  joblib.dump(rfc,'saved_var/rfc_binary')
else:
  rfc = joblib.load('saved_var/rfc_binary')
# Predict a sample:
# watch how many true predict in 100 sample 
# Take true predict      
for i in range(100): 
    sample_id = random.randint(1, 9999)
    if sgd_clf.predict([X_train[sample_id]]):
        print(sample_id)
        print(sgd_clf.predict([X_train[sample_id]]))
        
# Count amount true label = 6 on all-values in y_train_6
trueList = []
count = 0
for i in range(0, len(y_train_2) - 1):
  if(y_train_2[i][0]):
    trueList.append([y_train_2[i][0], count])
  count += 1
# Count = 5030 đúng (label = 2) trên 60000 sameple
# Take 100 position true and index of them
print(trueList[:100])



# In[3]: PERFORMANCE MEASURES 
# 3.1. Accuracy (with cross-validation) of SGDClassifier 
from sklearn.model_selection import cross_val_score
# Warning: takes time for new run! 
# accuracies là một độ đo (số lượng dự đoán đúng trên tất cả các sample) = right_predict / all(sample)
if new_run == True:
    accuracies = cross_val_score(sgd_clf, X_train, y_train_2, cv=3, scoring="accuracy")
    joblib.dump(accuracies,'saved_var/sgd_clf_binary_acc')
else:
    accuracies = joblib.load('saved_var/sgd_clf_binary_acc')



# 3.2. Accuracy of a dump classifier
# Note: We are having an IMBALANCED data, hence accuracy is not useful!
from sklearn.base import BaseEstimator
class DumpClassifier(BaseEstimator): # always return False (not-5 label)
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        # trả về mảng là những con số 0 hoặc 1 ở dạng boolean
        return np.zeros((len(X), 1), dtype=bool)
# những sameple không bằng 2 (áo mùa đông)
no_2_model = DumpClassifier()
# tính toán accuracies của Dump Model
# kết quả tính ra khá là cao [0.8966 0.9001 0.9033] mặc dù DumpClassifier không được trainning gì cã
# vì do dữ liệu của chêch lệch, như được đo ở dòng 104 (5030 / 54970) = 0.083
# 54970/60000 = 0.916 
# nó đoán ảnh False positive nhiều hơn True positive
cross_val_score(no_2_model, X_train, y_train_2, cv=3, scoring="accuracy")
'''
SGD:
[0.88805 0.794 0.91905]
Dump:
[0.8966 0.9001 0.9033]
Random:
Về accuracies thì random forest nhỉnh hơn và thời gian chạy của nó cũng lâu nhất
Vì nó có dạng ensemble
[0.9339 0.936  0.9372]

'''
# Note: 
#   >90% accuracy, due to only about 10% of the images are 5s.
#   IMBALANCED (or skewed) datasets: some classes are much more frequent than others.

# 3.3. Confusion matrix (better for imbalanced data)
# Info: number of times the classifier "confused" b/w samples of classes
from sklearn.model_selection import cross_val_predict 
# Warning: takes time for new run! 

if new_run == True:
    # Tính toán và trả về mảng cho tất cả prediction của từng sample 
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_2, cv=3)
    joblib.dump(y_train_pred,'saved_var/y_train_pred')
    y_train_pred_dump = cross_val_predict(no_2_model, X_train, y_train_2, cv=3)
    joblib.dump(y_train_pred,'saved_var/y_train_pred_dump')  
else:
    y_train_pred = joblib.load('saved_var/y_train_pred')
    y_train_pred_dump = joblib.load('saved_var/y_train_pred_dump')
# Tính confustion_matrix chỗ SDF và Dump
from sklearn.metrics import confusion_matrix  
conf_mat = confusion_matrix(y_train_2, y_train_pred) # row: actual class, column: predicted class. 
print(conf_mat)
conf_mat_dump = confusion_matrix(y_train_2, y_train_pred_dump) # row: actual class, column: predicted class. 
print(conf_mat_dump)
# Perfect prediction: zeros off the main diagonal 
y_train_perfect_predictions = y_train_2  # pretend we reached perfection
confusion_matrix(y_train_2, y_train_perfect_predictions)
'''
SGDClassifier
    [[47957  6043]
    [ 1935  4065]]
    
    [[54000     0]
    [ 6000     0]]
    array([[54000,     0],
        [    0,  6000]], dtype=int64)
RandomForestClassifier
    [[53731   269]
    [ 3589  2411]]
    (53731/60000) = 0.895 tỷ lệ âm tính thật cao hơn SGDClassifier, tuy nhiên
    tỷ lệ dự đoán Dương tính thật thấp hơn: 48.5% so với 80%
    array([[54000,     0],
        [    0,  6000]], dtype=int64)
'''
# SGD có kết quả khả quan vì số lượng trên đường chéo chính cao 47957/54970 = 0.87 (tỷ lệ đoán đúng đó không phải label 2) và 4065/5030 0.81 (tỷ lệ đoán đúng đó là label 2)
# Kết quả cho thấy mặt dù Dump có accuracies nhưng số lượng trện đường chéo chính (1, 1) = 0 => dự đoán chưa tốt
# True Possitive = 0, tỉ lệ đoán đúng label số 2 là 0%




# 3.4. Precision and recall (>> see slide)
from sklearn.metrics import precision_score, recall_score
# precision_score là tỷ lệ đoán đúng 1 sample đồ
# recall_score là tỷ lệ đoán đúng bao nhiêu % trên tập dữ liệu
print(precision_score(y_train_2, y_train_pred))
print(recall_score(y_train_2, y_train_pred))

"""
SGDC
0.40215670755836963 (cho thấy có 40% đoán đúng là ảnh đó thuộc hay không thuộc label 2 - áo mùa đông)
0.6775 (cho thấy trên 60000 sample thì đoán đúng được khoảng 68%) 60000*68% = 40800 dự đoán đúng là ảnh đó thuộc hay không thuộc label 2
RandomForest
0.8996268656716417 tỷ lệ dự đoán đúng gần 90% cho 1 sample nào đó
0.4018333333333333 cho thấy đúng 40% trên tổng số sameple tình nó khá 'nghiêm ngặt'

Kết luận: Model SGDC đánh đổi Precision để lấy tỉ lệ Recal và ngược lại đối với RandomForest
Dựa vào thông số trên em nghĩ rằng SGDC vẫn tốt hơn vì recall cao sẽ tìm được số trường hợp đúng dựa trên tổng số lượng
có thể áp dụng vào: phát hiện trộm, phát hiện vượt tốc độ, phát hiện thiên tai...
Còn với RandomForest sẽ tốt hơn khi áp dụng vào phát hiện nguồn ngước ngầm, phát hiện vàng... vì tính nghiêm ngặt của nó (chất lượng hơn số lượng)
"""


# 3.5. F1-score 
# Info: F1-score is the harmonic mean of precision and recall. 1: best, 0: worst.
# Đại diện cho hai đường precision và recall
# F1 = 2 × precision × recall / (precision + recall)
# F1 = 0.505 tỷ lệ tương gian của precision và recal chỉ khoảng 50% sự tác động qua lại của nó sẻ cân bằng
# F-score tốt hơn SGDC: 0.555
from sklearn.metrics import f1_score
f1_score(y_train_2, y_train_pred)


# 3.6. Precision/Recall tradeoff (>> see slide) 
# 3.6.1. Try classifying using some threshold (on score computed by the model)  
score = []
for i in range(0, 5):
  sample_id = random.randint(1000, 40000)
  y_score = sgd_clf.decision_function([X_train[sample_id]]) # score by the model
  threshold = 0
  y_some_digit_pred = (y_score > threshold)
  score.append([int(y_score[0]), sample_id, y_train_2[sample_id]])
print(score)

'''
    [Score, Index, Predict]
    Sameple có score càng lớn thì càng có nhiều khả năng thuộc class Positive và ngược lại

    [[-515, 7484, False], [-4733, 25409, False], [4906, 21259, True], [-2152, 18869, False], [-10460, 19088, False], 
    [-10887, 18137, False], [-11155, 5817, False], [-5450, 20220, False], [-5719, 24624, False], [3680, 19679, True], 
    [-7700, 25859, True], [-9889, 24229, False], [-13885, 3868, False], [-2029, 25239, True], [-19447, 22782, False], 
    [-6397, 4858, False], [1353, 32694, False], [-13569, 37370, False], [-6424, 31277, False], [-11752, 2929, False]]
'''

# Raising the threshold decreases recall

score = []
for i in range(0, 5):
  sample_id = random.randint(1000, 40000)
  y_score = sgd_clf.decision_function([X_train[sample_id]]) # score by the model
  threshold = 10000
  y_some_digit_pred = (y_score > threshold)
  score.append([int(y_score[0]), sample_id, y_train_2[sample_id]])
print(score)
# Khi ta để threshold càng cao thì, khả năng đoán trúng cao tuy nhiên recal khá thấp
 

# 3.6.2. Precision, recall curves wrt to thresholds 
# Get scores of all intances
# Warning: takes time for new run! 
if new_run == True:
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_2, cv=3, method="decision_function")
    joblib.dump(y_scores,'saved_var/y_scores')
else:
    y_scores = joblib.load('saved_var/y_scores')


# Plot precision,  recall curves
# precision_recall_curve lấy ra rât nhiều thresholds, mỗi thresholds sẽ lấy ra prediction và recall tương ứng
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_2, y_scores)
let_plot = True
if let_plot:
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend() 
    plt.grid(True)
    plt.xlabel("Threshold")   

'''
số liệu được tính trong hình thres_value_3000.png cho thấy với thres_value = 3000
ta có Presision = 0.467 và Recal = 0.537, nhìn chung xu hướng của đồ thị là 
thresold value càng cao thì Precision cao và ngược lại với Recall


Đồ thị được vẽ ở file Jupyter 19110148_NguyenNgocThien_Week05_RandomForestClassifier.ipynb
cho thấy mặc dù precision của Random tốt hơn SGDC từ đoạn giao điểm
'''

# Plot a threshold
thres_value = 3000
thres_id = np.min(np.where(thresholds >= thres_value))
precision_at_thres_id = precisions[thres_id] 
recall_at_thres_id = recalls[thres_id] 
if let_plot:
    plt.plot([thres_value, thres_value], [0, precision_at_thres_id], "r:")    
    plt.plot([thres_value], [precision_at_thres_id], "ro")                            
    plt.plot([thres_value], [recall_at_thres_id], "ro")            
    plt.text(thres_value+500, 0, thres_value)    
    plt.text(thres_value+500, precision_at_thres_id, np.round(precision_at_thres_id,3))                            
    plt.text(thres_value+500, recall_at_thres_id, np.round(recall_at_thres_id,3))     
    plt.savefig("thres_value_3000")   
    plt.show()

# 3.6.3. Precision vs recall curve (Precision-recall curve)
''' 
Dùng để dánh giá model 1 cách trực quan, bằng cách nhập hai đường lại
Nhìn chung đối với dữ liệu này phần lớn xu hướng có tính đi xuống -> hoạt động sẽ kém hiệu quả -> Classification chưa tốt

Đường cong của Random cao hơn SGDC -> hoạt động hiệu quả hơn -> Classification tốt

'''
if let_plot:         
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.title("Precision-recall curve (PR curve)")
    plt.savefig("Precision_recall_curv")   
    plt.show()


# 3.7. Receiver operating characteristic (ROC) curve 
# Info: another common measure for binary classifiers. 
# ROC curve: the True Positive Rate (= recall) against 
#   the False Positive Rate (= no. of false postives / total no. of actual negatives).
#   FPR is the ratio of negative instances that are incorrectly classified as positive.
# NOTE: 
#   Tradeoff: the higher TPR, the more FPR the classifier produces.
#   Good classifier goes toward the top-left corner.

# 3.7.1. Compute FPR, TPR for the SGDClassifier
from sklearn.metrics import roc_curve

# False Positive Rate (FPR) = FP / TN + FP - Tỉ lệ dương tính giả (số lượng chọn đúng giả)
# True Positive Rate - Tỉnh lệ dương tính thật (số lượng chọn label 2 đúng thật)
fpr, tpr, thresholds = roc_curve(y_train_2, y_scores)

# 3.7.2. Compute FPR, TPR for a random classifier (make prediction randomly)
from sklearn.dummy import DummyClassifier
dmy_clf = DummyClassifier(strategy="uniform")
y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_2, cv=3, method="predict_proba")
y_scores_dmy = y_probas_dmy[:, 1]
fprr, tprr, thresholdsr = roc_curve(y_train_2, y_scores_dmy)

# 3.7.3. Plot ROC curves
if let_plot:
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot(fprr, tprr, 'k--') # random classifier
    #plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal: random classifier
    plt.legend(['SGDClassifier','Random classifier'])
    plt.grid(True)        
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate (Recall)')    
    plt.show()

# 3.8. Compute Area under the curve (AUC) for ROC
# Info: 
#   A random classifier: ROC AUC = 0.5.
#   A perfect classifier: ROC AUC = 1.
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_train_2, y_scores)
    
# 3.9. ROC vs PR curve: when to use?
#   PR curve: focus the false positives (ie. u want high precision)
#   ROC: focus the false negatives (ie. u want high recall)
print('\n')


'''______DONE WEEK 05______'''

