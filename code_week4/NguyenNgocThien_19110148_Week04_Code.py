
'''
The following code is mainly from Chap 2, Géron 2019 
See https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb

LAST REVIEW: Feb 2022
'''
# Tham khao source code cua Thay

''' WEEK 03 '''


# In[0]: IMPORT AND FUNCTIONS
#region
from importlib.resources import path
from tabnanny import verbose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
from sklearn.model_selection import KFold   
import joblib 
#endregion


# In[1]: PART 1. LOOK AT THE BIG PICTURE (DONE)


# In[2]: PART 2. GET THE DATA (DONE). LOAD DATA
path = 'C:/Users/hp/Downloads/HK2 21-22/Machine Learning/CodeEx/code_week4/'
def ReadData(path):
    return pd.read_excel(path)

raw_data = ReadData(path + 'datasets/NguyenNgocThien_19110148_Week04_Data.xlsx')


# In[3]: PART 3. DISCOVER THE DATA TO GAIN INSIGHTS
#region
# 3.1 Quick view of the data
def Watch_Info_Data(raw_data): 
    
    print('\n____________ Dataset info ____________')
    print(raw_data.info())              
    print('\n____________ Some first data examples ____________')
    print(raw_data.head(10)) 
    
    print('\n____________ Counts on a feature ____________')
    league = raw_data['League'].value_counts()
    club = raw_data['Club'].value_counts()

    print(league) 
    print(club) 

    
    print('\n____________ Statistics of numeric features ____________')
    describe = raw_data.describe()
    print(describe)    
    
    print('\n____________ Get specific rows and cols ____________')     
    print(raw_data.loc[[0,5,20], ['Player', 'Value', 'Gls','League']] ) # Refer using column name
    print(raw_data.iloc[[0,5,20], [2, 8]] ) # Refer using column ID
  
# 3.2 Scatter plot b/w 2 features
def Show_Data_OneChart_Scatter(raw_data, x, y):
    raw_data.plot(kind="scatter", y=y, x=x, alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    plt.savefig(path +'/figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()      

def Show_Data_ManyCharts_Fix(raw_data, features):
    # 3.3 Scatter plot b/w every pair of features
    from pandas.plotting import scatter_matrix   
    features_to_plot = features
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig(path +'/figures/scatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

def Show_Data_Histogram(raw_data, features):
    # 3.4 Plot histogram of 1 feature
    from pandas.plotting import scatter_matrix   
    features_to_plot = features
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.show()

    # 3.5 Plot histogram of numeric features
    #raw_data.hist(bins=10, figsize=(10,5)) #bins: no. of intervals
    raw_data.hist(figsize=(12,8)) #bins: no. of intervals
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.savefig(path +'/figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
    plt.show()
    
corr_matrix = []
 # 3.6 Compute correlations b/w features
def Correlations_Feature(raw_data, index):
    corr_matrix = raw_data.corr()
    print(corr_matrix) # print correlation matrix
    print('\n',corr_matrix[index].sort_values(ascending=False)) # print correlation b/w a feature and other features

# 3.7 Try combining features

def Correlations_Combine_Feature(raw_data):
    raw_data["Value By Season"] = raw_data["Value"] / raw_data["Contract Years Left"] 
    raw_data["Gls Ast"] = raw_data["Gls"] + raw_data["Ast"] 
    corr_matrix = raw_data.corr()
    print(corr_matrix["Value By Season"].sort_values(ascending=False)) # print correlation b/w a feature and other features
    raw_data.drop(columns = ["Value By Season", "Gls Ast"], inplace=True) # remove experiment columns
    #endregion

# Print data    
    """
Watch_Info_Data(raw_data)
Show_Data_OneChart_Scatter(raw_data, "Value", "Gls")
feature = ["Value", "Age", "Gls", "Ast", "Total Mins/90"]
Show_Data_ManyCharts_Fix(raw_data, feature)
Show_Data_Histogram(raw_data, ["Value"])

    """




# In[4]: PART 4. PREPARE THE DATA 
#region
# 4.1 Remove unused features
# Red card va Yellow card khong anh huong nhieu
raw_data.drop(columns = ["Player", "Value Player", "Age"], inplace=True) 
 
#%% 4.2 Split training-test set and NEVER touch test set until test phase
method = 2
if method == 1: # Method 1: Randomly select 20% of data for test set. Used when data set is large
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42) # set random_state to get the same training set all the time, 
                                                                                     # otherwise, when repeating training many times, your model may see all the data
elif method == 2: # Method 2: Stratified sampling, to remain distributions of important features, see (Geron, 2019) page 56
    # Create new feature "KHOẢNG GIÁ": the distribution we want to remain
    raw_data["Interval Value"] = pd.cut(raw_data["Value"],
        bins=[0, 20000, 40000, 60000, 80000, 100000, np.inf],
        #labels=["<20 tr", "20-40 tr", "40-60 tr", "60-80 tr", "80-100 tr", ">100 tr"])
        labels=[2,4,6,8,10, 100]) # use numeric labels to plot histogram
    
    # Create training and test set
    from sklearn.model_selection import StratifiedShuffleSplit  
    splitter = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
    for train_index, test_index in splitter.split(raw_data, raw_data["Interval Value"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]      
    
    # See if it worked as expected
    if 0:
        raw_data["Interval Value"].hist(bins=6, figsize=(5,5)); plt.show();
        train_set["Interval Value"].hist(bins=6, figsize=(5,5)); plt.show()

    # Remove the new feature
    print(train_set.info())
    for _set_ in (train_set, test_set):
        #_set_.drop("income_cat", axis=1, inplace=True) # axis=1: drop cols, axis=0: drop rows
        _set_.drop(columns="Interval Value", inplace=True) 
    print(train_set.info())
    print(test_set.info())
    
print('\n____________ Split training and test set ____________')     
print(len(train_set), "training +", len(test_set), "test examples")
print(train_set.head(4))

#%% 4.3 Separate labels from data, since we do not process label values
train_set_labels = train_set["Value"].copy()
train_set = train_set.drop(columns = "Value") 
test_set_labels = test_set["Value"].copy()
test_set = test_set.drop(columns = "Value") 

#%% 4.4 Define pipelines for processing data. 
# INFO: Pipeline is a sequence of transformers (see Geron 2019, page 73). For step-by-step manipulation, see Details_toPipeline.py 

# 4.4.1 Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         

num_feat_names = ["MP", "Total Mins/90", "Gls", "Ast", "(G+A)/90", "Contract Years Left"] # =list(train_set.select_dtypes(include=[np.number]))
cat_feat_names = ['Club', 'Position', "League"] # =list(train_set.select_dtypes(exclude=[np.number])) 
# 4.4.2 Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
    ])   

trans_feat_values_1 = cat_pipeline.fit_transform(train_set)
# INFO: Try the code below to understand how a pipeline works

# 4.4.3 Define MyFeatureAdder: a transformer for adding features "TỔNG SỐ PHÒNG",...  
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_Gls_Ast = True): # MUST NO *args or **kargs
        self.add_Gls_Ast = add_Gls_Ast
    def fit(self, feature_values, labels = None):
        return self  # nothing to do here
    def transform(self, feature_values, labels = None):
        if self.add_Gls_Ast:        
            gls_id, ast_id = 1, 2 # column indices in num_feat_names. can't use column names b/c the transformer SimpleImputer removed them
            # NOTE: a transformer in a pipeline ALWAYS return dataframe.values (ie., NO header and row index)
            
            Total = feature_values[:, gls_id] + feature_values[:, ast_id]
            feature_values = np.c_[feature_values, Total] #concatenate np arrays
        return feature_values

# 4.4.4 Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), # copy=False: imputation will be done in-place 
    ('attribs_adder', MyFeatureAdder(add_Gls_Ast = True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
    ])  
  
# 4.4.5 Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  


# 4.5 Run the pipeline to process training data           
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(num_feat_names)))
joblib.dump(full_pipeline,path + r'models/full_pipeline_football.pkl')

#%% (optional) Add header to create dataframe. Just to see. We don't need header to run algorithms 
if 10: 
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    columns_header = train_set.columns.tolist() + ["add_Gls_Ast"] + onehot_cols
    for name in cat_feat_names:
        columns_header.remove(name)
    processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns = columns_header)
    print('\n____________ Processed dataframe ____________')
    print(processed_train_set.info())
    print(processed_train_set.head())
#endregion


''' WEEK 04 '''

# In[5]: PART 5. TRAIN AND EVALUATE MODELS 
#region
# 5.1 Try LinearRegression model
# 5.1.1 Training: learn a linear regression hypothesis using training data 
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(processed_train_set_val, train_set_labels)
print('\n____________ LinearRegression ____________')
print('Learned parameters: ', model.coef_)

# 5.1.2 Compute R2 score and root mean squared error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
        
# 5.1.3 Predict labels for some training instances
print("\nInput data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

# 5.1.4 Store models to files, to compare latter
#from sklearn.externals import joblib 
import joblib # new lib
def store_model(model, model_name = ""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,path +'models/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load(path +'models/' + model_name + '_model.pkl')
    #print(model)
    return model
store_model(model)


#%% 5.2 Try DecisionTreeRegressor model
# Training
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________ DecisionTreeRegressor ____________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.3 Try RandomForestRegressor model
# Training (NOTE: may take time if train_set is large)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________ RandomForestRegressor ____________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.4 Try polinomial regression model
# NOTE: polinomial regression can be treated as (multivariate) linear regression where high-degree features x1^2, x2^2, x1*x2... are seen as new features x3, x4, x5... 
# hence, to do polinomial regression, we add high-degree features to the data, then call linear regression
# 5.5.1 Training. NOTE: may take a while 
from sklearn.preprocessing import PolynomialFeatures
poly_feat_adder = PolynomialFeatures(degree = 2) # add high-degree features to the data
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name = "PolinomialRegression")      
else:
    model = load_model("PolinomialRegression")
# 5.4.2 Compute R2 score and root mean squared error
print('\n____________ Polinomial regression ____________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 5.4.3 Predict labels for some training instances
print("\nPredictions: ", model.predict(train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.5 Evaluate with K-fold cross validation 
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
#from sklearn.model_selection import cross_val_predict

#cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
#cv2 = StratifiedKFold(n_splits=10, random_state=42); 
#cv3 = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
print('\n____________ K-fold cross validation ____________')

run_evaluation = 1
if run_evaluation == 1:
    from sklearn.model_selection import KFold, StratifiedKFold
    # NOTE: 
    #   + If data labels are float, cross_val_score use KFold() to split cv data.
    #   + KFold randomly splits data, hence does NOT ensure data splits are the same (only StratifiedKFold may ensure that)
    cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator: just a try to persist data splits (hopefully)

    # Evaluate LinearRegression
    model_name = "LinearRegression" 
    model = LinearRegression()             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,path +'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate DecisionTreeRegressor
    model_name = "DecisionTreeRegressor" 
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,path +'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate RandomForestRegressor
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor(n_estimators = 5)
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,path +'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate Polinomial regression
    model_name = "PolinomialRegression" 
    model = LinearRegression()
    nmse_scores = cross_val_score(model, train_set_poly_added, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,path +'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load(path +'saved_objects/' + model_name + '_rmse.pkl')
    print("\nLinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load(path +'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load(path +'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "PolinomialRegression" 
    rmse_scores = joblib.load(path +'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
#endregion



# In[6]: PART 6. FINE-TUNE MODELS 
# NOTE: this takes TIME
#region
# IMPORTANT NOTE: since KFold split data randomly, the cv data in cross_val_score() above are DIFFERENT from SearchCV below.
#      => Should only compare resutls b/w SearchSV runs (NOT with cross_val_score()). 
# INFO: find best hyperparams (param set before learning, e.g., degree of polynomial in poly reg, no. of trees in rand forest, no. of layers in neural net)
# Here we fine-tune RandomForestRegressor and PolinomialRegression
print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

method = 1
# 6.1 Method 1: Grid search (try all combinations of hyperparams in param_grid)
if method == 1:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator
        
    run_new_search = 1      
    if run_new_search:        
        # 6.1.1 Fine-tune RandomForestRegressor
        model = RandomForestRegressor()
        param_grid = [
            # try 15 (3x4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 39]},
            # then try 12 (4x3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]} ]
            # Train across 5 folds, hence a total of (15+12)*5=135 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True) # refit=True: after finding best hyperparam, it fit() the model with whole data (hope to get better result)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,path +'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")      

        # 6.1.2 Fine-tune Polinomial regression          
        # model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()), # add high-degree features
        #                    ('lin_reg', LinearRegression()) ]) 
        # param_grid = [
        #     # try 3 values of degree
        #     {'poly_feat_adder__degree': [1, 2, 3]} ] # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
        #     # Train across 5 folds, hence a total of 3*5=15 rounds of training 
        # grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
        # grid_search.fit(processed_train_set_val, train_set_labels)
        # joblib.dump(grid_search,path +'saved_objects/PolinomialRegression_gridsearch.pkl') 
        # print_search_result(grid_search, model_name = "PolinomialRegression") 
    else:
        # Load grid_search
        grid_search = joblib.load(path +'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        grid_search = joblib.load(path +'saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name = "PolinomialRegression") 

# 6.2 Method 2: [EXERCISE] Random search n_iter times 
elif method == 2:
     # Exercise
    model = LinearRegression();
    from scipy.stats import loguniform 
    from scipy.stats import loguniform_int 
    param_distributions = {
        'classifier__l2_regularization': loguniform(1e-6, 1e3),
        'classifier__learning_rate': loguniform(0.001, 10),
        'classifier__max_leaf_nodes': loguniform_int(2, 256),
        'classifier__min_samples_leaf': loguniform_int(1, 100),
        'classifier__max_bins': loguniform_int(2, 255),
    }
    model_random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter = 100, cv=5, verbose=1)
    model_random_search.fit(processed_train_set_val, train_set_labels)

     

     
#endregion



# In[7]: PART 7. ANALYZE AND TEST YOUR SOLUTION
# NOTE: solution is the best model from the previous steps. 
#region
# 7.1 Pick the best model - the SOLUTION
# Pick Random forest
searchs = joblib.load(path +'/saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = searchs.best_estimator_
# Pick Linear regression
#best_model = joblib.load('saved_objects/LinearRegression_model.pkl')

print('\n____________ ANALYZE AND TEST YOUR SOLUTION ____________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUTION")   

# 7.2 Analyse the SOLUTION to get more insights about the data
# NOTE: ONLY for rand forest
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + ["add_Gls_Ast"] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

# 7.3 Run on test data
full_pipeline = joblib.load(path +'/models/full_pipeline_football.pkl')
processed_test_set = full_pipeline.fit_transform(test_set)  
best_model.fit(processed_test_set, test_set_labels)
# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("\nPredictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')


#endregion



# In[8]: PART 8. LAUNCH, MONITOR, AND MAINTAIN YOUR SYSTEM
# Go to slide: see notes

done = 1

# Time 1

    # Performance on test data:
    # R2 score (on test data, best=1): 0.8741313933022942
    # Root Mean Square Error:  6156.4
    # Predictions:  [13665. 33810. 16500. 15375. 10845. 42840. 11835. 28800. 15390.]
    # Labels:       [13500, 19800, 18000, 9000, 9000, 49500, 10800, 31500, 9000]
    
# Time 2
    # Performance on test data:
    # R2 score (on test data, best=1): 1.0
    # Root Mean Square Error:  0.0
    # Predictions:  [13500. 19800. 18000.  9000.  9000. 49500. 10800. 31500.  9000.]
    # Labels:       [13500, 19800, 18000, 9000, 9000, 49500, 10800, 31500, 9000]

#Time 3
    # Performance on test data:
    # R2 score (on test data, best=1): 1.0
    # Root Mean Square Error:  0.0
    # Predictions:  [13500. 19800. 18000.  9000.  9000. 49500. 10800. 31500.  9000.]
    # Labels:       [13500, 19800, 18000, 9000, 9000, 49500, 10800, 31500, 9000]
    
#Time 4
    # Performance on test data:
    # R2 score (on test data, best=1): 0.88547896234333
    # Root Mean Square Error:  5872.4
    # Predictions:  [13755. 21300. 17310. 12630. 11760. 38250. 11475. 24708. 10365.]
    # Labels:       [13500, 19800, 18000, 9000, 9000, 49500, 10800, 31500, 9000]
    
#Time 5
    # Performance on test data:
    # R2 score (on test data, best=1): 0.8954612198286253
    # Root Mean Square Error:  5610.6
    # Predictions:  [13245. 23880. 17565. 15240. 14415. 45930. 12990. 27705. 11490.]
    # Labels:       [13500, 19800, 18000, 9000, 9000, 49500, 10800, 31500, 9000] 