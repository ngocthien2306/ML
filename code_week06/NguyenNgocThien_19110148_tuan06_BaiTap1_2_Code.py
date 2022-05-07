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
rfc = RandomForestClassifier(n_estimators=70, oob_score=True, random_state=101)
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
        
# Count amount true label = 2 on all-values in y_train_2
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


# In[5]: MULTICLASS CLASSIFICATION (>> see slide)
# 5.1. Try SGDClassifier 
# Info: SkLearn automatically runs OvA when you try to 
#       use a binary classifier for a multiclass classification.
# Warning: takes time for new run! 
'''
    Training lại dữ liệu với max_iterate là 100 -> model sẽ tốt hơn vì được lập lại nhiều lần. Hệ quả -> tốn thời gian hơn
'''
if new_run == True:
    from sklearn.linear_model import SGDClassifier       
    sgd_clf = SGDClassifier(random_state=42, max_iter=100) # set random_state to reproduce the result
    sgd_clf.fit(X_train, y_train) # y_train, not y_train_2
    joblib.dump(sgd_clf,'saved_var/sgd_clf_multi')
else:
    sgd_clf = joblib.load('saved_var/sgd_clf_multi')

# Training model RandomForestClassifier, với n_estimators là số lượng số cần lấy trên 1 tập dữ liệu
     
randomForest = True
# RandomForestClassifier
if randomForest:
    rfc = RandomForestClassifier(n_estimators=70, oob_score=True, random_state=101)
    if new_run:
        rfc.fit(X_train, y_train)
        joblib.dump(rfc,'saved_var/rfc_binary')
    else:
        rfc = joblib.load('saved_var/rfc_binary')
# kiểm tra tỷ lệ dự đoán đúng trên 100 lần đoán
# tỷ lệ vào khoảng 76 - 88 trên 100 lần đoán và trung bình khoảng 82%
count = 0
for i in range(100):
  sample_id = random.randint(1, 60000)
  predict = sgd_clf.predict([X_train[sample_id]])
  label = y_train[sample_id]
  if predict == label:
    count += 1
print(count)
'''
    RandomForestClassifier
    kiểm tra tỷ lệ dự đoán đúng trên 100 lần đoán
    tỷ lệ hầu như là 100 trên 100 lần đoán và trung bình tuyệt đối 100%
    -> tốt hơn SGD
'''
if randomForest:
    count = 0
    for i in range(100):
        sample_id = random.randint(1, 60000)
        predict = rfc.predict([X_train[sample_id]])
        label = y_train[sample_id]
        if predict == label:
            count += 1
    print(count)
  
# Try prediction
sample_id = 15
print(sgd_clf.predict([X_train[sample_id]]))
print(y_train[sample_id])
# To see scores from classifers
print(sgd_clf.classes_)
sample_scores = sgd_clf.decision_function([X_train[sample_id]]) 
print(sample_scores)
class_with_max_score = np.argmax(sample_scores)

# Try prediction
if randomForest:
    sample_id = 15
    print(rfc.predict([X_train[sample_id]]))
    print(y_train[sample_id])
    # To see scores from classifers
    print(rfc.classes_)
    sample_scores = rfc.predict_proba([X_train[sample_id]]) 
    print(sample_scores)
    class_with_max_score = np.argmax(sample_scores)
'''
    [9] - số dự đoán
    9 - label 
    [0 1 2 3 4 5 6 7 8 9] - các class được training trong model, kiểm tra xem có bị thiếu sót label nào cần training hay không?
    [[ -70375.38111943  -80421.17179755  -63193.89185674  -98200.2008379
    -146364.53775501  -20055.27125229  -35792.47198095  -35893.26816411
    -65500.26628055   12958.13892661]]
    Với RandomForestClassifier con số sẽ hiển thị dưới dạng phần trăm (xác xuất đoán đúng)
    như ta có % cao nhất là 0.97 = 97% là ở số 9 
    [[0.   0.   0.   0.   0.   0.
      0.   0.02857143 0.   0.97142857]]
   

'''

# 5.2. Force sklearn to run OvO (OneVsOneClassifier) or OvA (OneVsRestClassifier)
from sklearn.multiclass import OneVsRestClassifier
# Warning: takes time for new run! 
if randomForest:
    ova_clf = OneVsRestClassifier(RandomForestClassifier(random_state=101, n_estimators=70))
    if new_run == True:
        ova_clf.fit(X_train, y_train)
        joblib.dump(ova_clf,'saved_var/ova_random_clf')
    else:
        ova_clf = joblib.load('saved_var/ova_random_clf')
    print(len(ova_clf.estimators_))
    sample_scores = ova_clf.predict_proba([X_train[sample_id]]) 
    print(sample_scores)

ova_clf = OneVsRestClassifier(SGDClassifier(random_state=42))
if new_run == True:
    ova_clf.fit(X_train, y_train)
    joblib.dump(ova_clf,'saved_var/ova_clf')
else:
    ova_clf = joblib.load('saved_var/ova_clf')
print(len(ova_clf.estimators_))
sample_scores = ova_clf.decision_function([X_train[sample_id]]) 
print(sample_scores)

'''
    10 -  độ dài của estimators_ 
    hàm decision_function này sẽ tính điểm các class cho từng sample, dự vào model multi OneVsRestClassifier với SGDClassifier là base class
    [[-32042.04487256 -70900.83114438 -38973.74845563 -66200.92051102
    -72819.23822588 -19747.29000385 -17659.19120913 -19924.50119551
    -44236.58193342  14898.37652544]]
    sample_scores: tính toán điểm của từng class nếu class nào có điểm cao hơn thì kết quả chính xác thuộc về class đó
    theo ví dụ trên được dự đoán là số 9 -> đối chiếu với mảng điểm trên, ta nhận thấy rằng phần tử 09 có điểm cao nhất -> thuộc về class 9
    
    với RandomForestClassifier vs OneVsRestClassifier ta dùng hàm predict_proba để tính điểm (xác xuất của mỗi label)
    [[0.         0.         0.         0.         0.         0.
    0.         0.01428571 0.         0.98571429]]
    
    Kết luận: qua OneVsRestClassifier ta có thể thấy điểm decision_function với SGD model và xác xuất predict_proba với Random model
    Có phần cao hơn vì thực hiện multi classification
    12958 - 14898
    0.97 - 0.986
    
    Multi classification với 2 model này kết quả tốt hơn so với bình thường. Tuy nhiên em vẫn đánh giá RandomForestClassifier tốt hơn vì độ chính xác cao hơn
    Nhược điểm là traning model sẽ lâu hơn SGD, mặc dù vậy khi chạy thực tế người ta vẫn yêu cầu độ chính xác cao (vì tính cấp thiết ứng dụng của nó: dự đoán thiên tai, ăn trộm, phạm luật)
    -> cần phải đánh đổi thời gian training
   
'''

from sklearn.multiclass import OneVsOneClassifier
if randomForest:
    ovo_clf = OneVsOneClassifier(RandomForestClassifier(random_state=101, n_estimators=70))
    if new_run == True:
        ovo_clf.fit(X_train, y_train)
        joblib.dump(ovo_clf,'saved_var/ovo_clf')
    else:
        ovo_clf = joblib.load('saved_var/ovo_clf')
    print(len(ovo_clf.estimators_))
    sample_scores = ovo_clf.decision_function([X_train[sample_id]]) 
    print(sample_scores)
# Warning: takes time for new run! 
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
if new_run == True:
    ovo_clf.fit(X_train, y_train)
    joblib.dump(ovo_clf,'saved_var/ovo_clf')
else:
    ovo_clf = joblib.load('saved_var/ovo_clf')
print(len(ovo_clf.estimators_))
sample_scores = ovo_clf.decision_function([X_train[sample_id]]) 
print(sample_scores)
'''
    45 vì là chọn 2 số trong 10 từ 0 đến 9 số với điều kiện là không chọn số đã chọn trước -> ta có công thức như sau N(N-1)/2 cách chọn 
    vì có 0 : 9 class -> 10(10-1)/2 = 45
    
    [[ 1.66666677 -0.33333329  3.66666684  0.66666674  2.66666676  8.33333328
    4.66666692  7.33333328  6.33333299  9.33333329]]
    Tổng tất cả các số này sấp xĩ 45 -> nếu số làm có lượt bình chọn cao nhất -> khả năng đúng cao nhất
    Theo ví dụ trên ta có xấp xỉ 9 lượt bình chọn cho số 9 vì vậy khả năng dự đoán đó là số 9 cao hơn
    Hoàn toàn đúng với các ví dụ trên là model OneVsRestClassifier multi class, model SGDClassifier 
    
    RandomForestClassifier vs OneVsOneClassifier
    45
    [[ 3.71370968 -0.29151732  1.7287234   0.74097665  2.76767677  7.25476992
    5.06818182  8.28379335  6.27764519  9.3       ]]
    
    Đối chiếu với RandomForestClassifier vs SGD ta thấy rằng các lượt bình chọn của 2 model này xấp xĩ nhau
    Những lượt bình chọn âm xấp xỉ = 0
    
    Kết luận chung: với OneVsOneClassifier thì 2 model có sự tiến triển hơn (xấp xĩ bằng nhau) 
'''

# In[6]: EVALUATE CLASSIFIERS
# 6.1. SGDClassifier  
# Warning: takes time for new run! 
# Tính điểm accuracy với model SGDClassifier
if new_run == True:
    sgd_acc = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    joblib.dump(sgd_acc,'saved_var/sgd_acc_multi')
else:
    sgd_acc = joblib.load('saved_var/sgd_acc_multi')
print(sgd_acc)

# 6.2. RandomForestClassifier  
# Warning: takes time for new run! 
if new_run == True:
    forest_acc = cross_val_score(rfc, X_train, y_train, cv=3, scoring="accuracy")
    joblib.dump(forest_acc,'saved_var/forest_acc_multi')
else:
    forest_acc = joblib.load('saved_var/forest_acc_multi')
print(forest_acc)
'''
So sánh SGDClassifier với RandomForestClassifier
    SGDClassifier: [0.78315 0.81355 0.82255] - tỷ lệ đoán đúng trung bình khoảng 80% gần bằng với dự đoán (tính toán theo xác xuất) ở hàng 421
    RandomForestClassifier: [0.87455 0.8801  0.87755] - tỷ lệ trung bình khoảng 88% -> cao hơn SGDClassifier -> model tốt hơn vì non-linear dùng đường parapol hoặc hyperpol để dự đoán dữ liệu
    Tuy con số này sai nhiều so với tính toán theo xác xuất ở dòng 425
'''

# In[7]: SCALE FEATURES AND EVALUATE CLASSIFIERS AGAIN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# 7.1. SGDClassifier (benefited from feature scaling)
# Warning: takes time for new run! 
if new_run == True:
    sgd_acc_after_scaling = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=4)
    joblib.dump(sgd_acc_after_scaling,'saved_var/sgd_acc_after_scaling')
else:
    sgd_acc_after_scaling = joblib.load('saved_var/sgd_acc_after_scaling')
print(sgd_acc_after_scaling)
# 7.2. RandomForestClassifier  
# Warning: takes time for new run! 
if new_run == True:
    forest_acc_after_scaling = cross_val_score(rfc, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=4)
    joblib.dump(forest_acc_after_scaling,'saved_var/forest_acc_after_scaling')
else:
    forest_acc_after_scaling = joblib.load('saved_var/forest_acc_after_scaling')
print(forest_acc_after_scaling)

'''
SGDClassifier: [0.83035 0.83545 0.83725]
RandomForestClassifier: [0.8746  0.88005 0.8778 ]
Nhìn chung sau khi scale thì SGDClassifier đã được cải thiện nhiều hơn, có thể do thu gọn miền giá trị 
-> đường thẳng biểu diễn đi qua được nhiều điểm dữ liệu hơn -> mô tả chung về dữ liệu chính xác hơn -> tỷ lệ tăng
Với RandomForestClassifier hầu như không có sự thây đổi bản bất của model này là đường hyperol
-> đường cong này vẫn mô tả đúng được các dữ liệu khi miền giá trị bị co lại 
'''
# In[8]: ERROR ANALYSIS 
# NOTE: Here we skipped steps (eg. trying other data preparation options, hyperparameter tunning...)
#       Assumming that we found a promissing model, and are trying to improve it.
#       One way is to analyze errors it made.

# 8.1. Plot confusion matrix
# Warning: takes time for new run! 
'''
RandomForestClassifier
'''
if randomForest:
    if new_run == True:
        y_train_pred = cross_val_predict(rfc, X_train_scaled, y_train, cv=3)
        joblib.dump(y_train_pred,'saved_var/y_train_pred_random')
    else:
        y_train_pred = joblib.load('saved_var/y_train_pred_random')
    conf_mat = confusion_matrix(y_train, y_train_pred) # row: actual class, col: prediction
    let_plot = True;
    if let_plot:
        plt.matshow(conf_mat, cmap=plt.cm.seismic)
        plt.xlabel("Prediction")
        plt.ylabel("Actual class")
        plt.colorbar()
        plt.savefig("figs/confusionRandom_matrix_plot")
        plt.show()
        
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_conf_mat = conf_mat / row_sums
    # Replace rates on diagonal (correct classifitions) by zeros    
    if let_plot:
        np.fill_diagonal(norm_conf_mat, 0)
        plt.matshow(norm_conf_mat,cmap=plt.cm.seismic)
        plt.xlabel("Prediction")
        plt.ylabel("Actual class")
        plt.colorbar()
        plt.savefig("figs/confusion_matrix_errors_plot", tight_layout=False)
        plt.show()
        
if new_run == True:
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    joblib.dump(y_train_pred,'saved_var/y_train_pred_step8')
else:
    y_train_pred = joblib.load('saved_var/y_train_pred_step8')
conf_mat = confusion_matrix(y_train, y_train_pred) # row: actual class, col: prediction
let_plot = True;
if let_plot:
    plt.matshow(conf_mat, cmap=plt.cm.seismic)
    plt.xlabel("Prediction")
    plt.ylabel("Actual class")
    plt.colorbar()
    plt.savefig("figs/confusionSGD_matrix_plot")
    plt.show()

# 8.2. Plot error-only confusion matrix 
# Convert no. of intances to rates
row_sums = conf_mat.sum(axis=1, keepdims=True)
norm_conf_mat = conf_mat / row_sums
# Replace rates on diagonal (correct classifitions) by zeros    
if let_plot:
    np.fill_diagonal(norm_conf_mat, 0)
    plt.matshow(norm_conf_mat,cmap=plt.cm.seismic)
    plt.xlabel("Prediction")
    plt.ylabel("Actual class")
    plt.colorbar()
    plt.savefig("figs/confusion_matrix_errors_plot", tight_layout=False)
    plt.show()

'''
SGDClassifier
Dữ liệu trên được biểu diện dưới dạng độ thị như hình figs\confusionSGD_matrix_plot
Nhận thấy rằng con số trên đường chéo chính khá cao, tuy nhiên vẫn có những label bị đoán sai nhiều
array([[4806,   19,  111,  514,   22,    1,  429,    0,   98,    0],
       [  12, 5688,   47,  198,   16,    1,   35,    0,    3,    0],
       [  46,    5, 4461,  126,  766,    1,  521,    1,   73,    0],
       [ 211,  106,   72, 5249,  197,    0,  150,    0,   15,    0],
       [   8,    4,  676,  354, 4394,    0,  536,    1,   27,    0],
       [   3,    3,    2,   13,    2, 5520,    6,  256,   69,  126],
       [ 806,   19,  700,  435,  533,    0, 3311,    1,  194,    1],
       [   0,    0,    0,    0,    0,  353,    0, 5390,   15,  242],
       [  31,    3,   29,  129,   37,   11,   97,   35, 5622,    6],
       [   0,    3,    0,    6,    0,  106,    2,  249,    4, 5630]], dtype=int64)
       
Dữ liệu trên được biểu diện dưới dạng độ thị như hình figs\confusionRandom_matrix_plot      
RandomForestClassifier       
array([[5148,    2,   88,  198,   20,    4,  490,    0,   50,    0],
       [  10, 5783,   23,  135,    6,    1,   40,    0,    2,    0],
       [  34,    2, 4865,   62,  666,    1,  332,    0,   38,    0],
       [ 133,   21,   50, 5483,  174,    0,  126,    0,   13,    0],
       [  12,    7,  477,  228, 4969,    1,  285,    0,   20,    1],
       [   0,    0,    1,    1,    0, 5727,    1,  178,   27,   65],
       [ 958,    3,  703,  149,  543,    4, 3544,    0,   96,    0],
       [   0,    0,    0,    0,    0,  105,    0, 5610,   11,  274],
       [   9,    2,   22,   20,   29,   19,   71,   11, 5813,    4],
       [   0,    0,    0,    1,    1,   70,    5,  208,    8, 5707]],
      dtype=int64)
       
Dựa vào 2 số liệu trên ta có thể kết luận: các số trên đường chéo chính của RandomForestClassifier luôn lớn hơn SGDClassifier -> cho thấy model dự đoán dương tính thật khá tốt
Nhưng ở chiều hướng ngược lại max(error) = 958 > 806 của SGDClassifier. Số class 6 nhầm lẫn thành class 0 cao
-> cần nhanh chóng khắc phục lỗi này bằng các biện pháp đã nêu ở dưới (780)

Kết luận: RandomForestClassifier vẫn tốt hơn SGDClassifier vì số lượt các món thời trang bị nhầm lẫn giảm đáng kể ở các vị trí khác 
như là 4:2, 0:3, 4:3, 6:3, etc. Vì vậy tổng thể tốt hơn

'''
# 8.3. What to do with class 8? (>> see slide)
# Trả lời câu hỏi cho trường hợp trong slide, tương tự như đối với trường hợp này với bộ dữ liệu fashion
"""
Khi vẽ đồ thị dựa trên tỷ lệ giữa 1 cột trên tổng cột trên 1 hàng và cho đường chéo chính bằng 0
thì cột nào có màu đậm thì cột đó dự đoán sai nhiều nhất và cần phải được cải thiện ngay vị trí đó
Ví dụ trên đối với class 8 nó có nhiều số dự đoán thành số 8 nhiều nhất, vì vậy ta sẻ chú trọng vào class 8 
để cải thiện và cụ thể là các số như 5 4 9 2 3 là số phần trăm dự đoán thành số 8 nhiều nhất
Gồm các cách cải thiện như sau
    1. thêm dữ liệu cho các class bị sai nhiều như là 5 4 9 2 và 3 để cho model có thêm thông tin phân biệt 
    với class số 8 và các class còn lại -> huấn luyện lại
    2. xử lí ảnh cho các class, vì nguyên nhân có thể do ảnh còn mờ và đứt nét nên khả năng học các đặc trưng 
    có thể bị sai lệch cao.
    Ví dụ ta có thể dùng các thuật toán như erosion dilation (tăng sáng cho các vùng sáng) cần phải chuyển sang dạng âm bảng
    vì khi tăng sáng thì các số mới có thể rỏ sáng hơn giúp cho dễ dàng nhận biết các đặc trưng
    3. xác định các đường nét của các số ví dụ số 1 ta chỉ cần 1 nét để vẽ, số 4 thì 2 đường,...
"""
# 8.4. Plot examples of 3s and 5s
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
'''

    label 0: áo thun tay ngắn, cổ trònnhóm phụ kiện này liên quan tới áo thun
    label 1: thuộc nhóm quần jean
    label 2: áo thun mùa đông (sweater), ôm cổ, tay dài
    label 3: liên quan đến nhóm phụ kiện váy dài ôm body
    label 4: là nhóm phụ kiện thuộc áo khoác dài tay
    label 5: là các phụ kiện thời trang thuộc loại giày sandal
    label 6: nhóm phụ kiện này liên quan tới áo thun
    label 7: phụ kiện thời trang thuộc loại giày adidas 
    label 8: các phụ kiện liên quan đến túi sách
    label 9: giày cao cổ (boot)

    Dựa vào dữ liệu được đánh giá ở trên bằng hàm confusion_matrix và biểu diễn dưới dạng đồ thị ta có thể thấy rằng các label
    như 0 2 3 4 6 bị nhầm lẫn nhiều nhất vì các đặc trưng khá giống nhau, khó phân biệt
    Vì các nhóm này thuộc nhóm phụ kiện áo như áo thu, áo khoác, váy -> dễ bị nhầm lẫn với nhau
    Cụ thể là 6:0, 4:2, 6:2, 0:3, 4:3, 6:3,... etc
    Lỗi (dự đoán sai) sẽ xảy ra cao nhất ở các nhóm này -> sử dụng các phương pháp như đã mô tả ở trên đối với class 8
        1. thêm dữ liệu cho các class bị sai nhiều như là 5 4 9 2 và 3 để cho model có thêm thông tin phân biệt 
        với class số 8 và các class còn lại -> huấn luyện lại
        2. xử lí ảnh cho các class, vì nguyên nhân có thể do ảnh còn mờ và đứt nét nên khả năng học các đặc trưng 
        có thể bị sai lệch cao.
        Ví dụ ta có thể dùng các thuật toán như erosion dilation (tăng sáng cho các vùng sáng) cần phải chuyển sang dạng âm bảng
        vì khi tăng sáng thì các số mới có thể rỏ sáng hơn giúp cho dễ dàng nhận biết các đặc trưng
        3. xác định các đường nét của các kiểu áo ví dụ áo thun ôm với áo thun rộng,...
"""
'''
class_A = 0
class_B = 6
X_class_AA = X_train[(y_train == class_A) & (y_train_pred == class_A)]
X_class_AB = X_train[(y_train == class_A) & (y_train_pred == class_B)]
X_class_BA = X_train[(y_train == class_B) & (y_train_pred == class_A)]
X_class_BB = X_train[(y_train == class_B) & (y_train_pred == class_B)] 
    
if let_plot:
    plt.figure(figsize=(6,7))
    plt.subplot(221); plot_digits(X_class_AA[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_A) + ", Predicted: " + str(class_A))
    plt.subplot(222); plot_digits(X_class_AB[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_A) + ", Predicted: " + str(class_B))
    plt.subplot(223); plot_digits(X_class_BA[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_B) + ", Predicted: " + str(class_A))
    plt.subplot(224); plot_digits(X_class_BB[0:25], images_per_row=5)
    plt.title("Actual: " + str(class_B) + ", Predicted: " + str(class_B))
    plt.show()
'''
    Hiển thị các hình ảnh của class 0 và 6 để có thể hiểu hơn vì sao model bị nhầm lẫn nhiều về các đặc trưng
    0. áo thun tay ngắn, cổ tròn
    6. nhóm phụ kiện  liên quan tới áo thun
    Vì tính chất của 2 phụ kiện này khá giống nhau -> khó phân biệt 
    Điều này sẽ xảy ra cao hơn với SGD model -> cải thiện cần sử dụng RandomForest để mô tả dữ liệu
'''

# 8.5. What hapened with 3s and 5s images? (>> see slide)
print('\n')
# Trả lời câu hỏi cho trường hợp trong slide, tương tự như đối với trường hợp này với bộ dữ liệu fashion
'''
    Trong trường hợp này đối với số 3 và 5 trên thực tế nét vẽ khá là giống nhau -> các đặc trưng giống nhau -> khó phân biệt
    Sử dụng model SGD sẽ làm giảm thêm khả năng dự đoán đúng vì nó là linear model -> không mô tả toàn bộ dữ liệu bằng 1 được thẳng được
    Do dự giống nhau và tính chất đặt thù của model -> 5s và 3s sẽ dễ bị dự đoán sai nhiều hơn
'''

# In[10]: MULTILABEL CLASSIFICATION (Multi-[binary] label) 
# Info: (>> see slide) 
# 10.1. Create multilabel labels
# tạo ra nhiều label để training, điều này sẽ giúp model có nhiều thông tin hơn -> dự đoán đúng cao
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

# 10.2. Try KNeighborsClassifier    
# Note: KNeighborsClassifier supports multilabel classification. Not all classifiers do. 
# Sử dụng KNeighborsClassifier để thực hiện muti classify, label dưới dạng binary vector
from sklearn.neighbors import KNeighborsClassifier
# Warning: takes time for new run! 
knn_clf = KNeighborsClassifier()
if new_run == True:
    knn_clf.fit(X_train, y_multilabel)
    joblib.dump(knn_clf,'saved_var/knn_clf')
else:
    knn_clf = joblib.load('saved_var/knn_clf')
# Try prediction trên multi classify
sample_id = 14652;
print(knn_clf.predict([X_train[sample_id]]))
print(y_train[sample_id])
print(y_multilabel[sample_id])
''' với id = 14652 ta có label bằng 2, dựa vào model mới traning lại ta xác định rằng 2 < 7 và là even number
    [[False False]]
    2
    [False False]
'''
 
# 10.3. Evaluate a multilabel classifier
# Note: many ways to do this, e.g., measure the F1 score for each individual label then compute the average score
# WARNING: may take HOURS for a new run! 
if new_run == True:
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    joblib.dump(y_train_knn_pred,'saved_var/y_train_knn_pred')
else:
    y_train_knn_pred = joblib.load('saved_var/y_train_knn_pred')
f1_score(y_multilabel, y_train_knn_pred, average="macro") # macro: unweighted mean, weighted: average weighted by support (no. of true instances for each label)
'''
    Tính performace measure trên tập label
    Trung bình cộng của các label >= hoặc là even or old number
    f1_core = 0.966
    Kết luận: nhờ thực hiện multi classify -> model đã cải thiện được độ chính xác hơn
'''

# In[11]: MULTIOUTPUT CLASSIFICATION (>> see slide) 
# 11.1. Add noise to data
# Create noisy features

'''
    MULTIOUTPUT: ngoài cho biết sample bất kì thuộc class nào nó còn đưa ra nhiều thông tin về label đó
    Label (vector) [4, 0, 7.8, 8.5, 0] mỗi con số này sẽ có ý nghĩa khác nhau
'''
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
'''
    Tạo nhiễu (noisy) random các số từ 0 -> 100, sau đó cộng với X_train -> label nhiễu (các đóm màu xung quanh) 
'''
# Labels now are clear images
y_train_mod = X_train
y_test_mod = X_test
# Plot a sample and its label
if let_plot:
    sample_id = 0
    plt.subplot(121); 
    plot_digit(X_train_mod[sample_id],str(y_train[sample_id])+" (noisy FEATURE)",showed=False)
    plt.subplot(122); 
    plot_digit(y_train_mod[sample_id],str(y_train[sample_id])+" (LABEL)",showed=True)
'''
    Đưa vào model 1 tấm ảnh nhiễu -> dùng classification để khử nhiễu
    Mỗi label sẽ có 28x28 = 784 giá trị pixel
'''
# 11.2. Training
# Warning: takes time for a new run! 
'''
    Đưa vào tập dữ liệu nhiễu ở bên trên để thực hiện training classification model
'''
if new_run == True:
    knn_clf.fit(X_train_mod, y_train_mod)
    joblib.dump(knn_clf,'saved_var/knn_clf_multioutput')
else:
    knn_clf = joblib.load('saved_var/knn_clf_multioutput')

# 11.3. Try predition
sample_id = 12
clean_digit = knn_clf.predict([X_test_mod[sample_id]])    
if let_plot:
    plt.figure(figsize=[12,5])
    plt.subplot(131); 
    plot_digit(X_test_mod[sample_id],str(y_test[sample_id])+" (input SAMPLE)",showed=False)
    plt.subplot(132); 
    plot_digit(clean_digit,str(y_test[sample_id])+" (PREDICTION)",showed=False)
    plt.subplot(133); 
    plot_digit(y_test_mod[sample_id],str(y_test[sample_id])+" (LABEL)",showed=True)

'''
    Thực hiện predition trên tập nhiễu đã được training, nhận thấy rằng các điểm nhiễu đã được làm rõ
    Tuy nhiên chưa rỏ bằng ảnh gốc ban đầu, nhưng nhờ vậy nó đã tạo ra được nhiều đặc trưng cũng như multi output của mỗi tấm ảnh 
'''



# Câu 2:
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# accuracy_score 0.972 với n_neighbors = [3, 4, 5] và weights = ['uniform', 'distance']
# best_params_ là n_neighbors = 4 và n_neighbors = distance
param_grid = {'n_neighbors': [3, 4, 5], 'weights': ['uniform', 'distance']}
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)

y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))
# End of our classification tour.






 