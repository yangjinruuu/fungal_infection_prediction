import warnings
warnings.filterwarnings("ignore")
import os
import pickle
from scipy.stats import norm
import numpy as np
import pandas as pd
from tableone import TableOne
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from missforest import MissForest
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import shap
import re
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier   # 新增

os.getcwd() # 当前工作路径
data = pd.read_csv("分析数据83英文.csv")

print(data.head())  # 查看前五行数据
print(data.info()) 

data['Occupation'] = data['Occupation'].astype('category')
data['Disease type'] = data['Disease type'].astype('category')
data['Admission route'] = data['Admission route'].astype('category')

###############################################################################
#################################### 数据拆分 ##################################
###############################################################################
##################### 1. 数据集拆分：训练集和测试集 ######################

train_data, test_data = train_test_split(
    data, 
    test_size=0.3,
    stratify=data["Fungal infection"], 
    random_state=2025
)

# 保存这两个数据集
train_data.to_csv("train_data_notscaled.csv", index=False)
test_data.to_csv("test_data_notscaled.csv", index=False)

######################## 2. 查看训练集vs测试集的变量均衡性 ########################
train_data["group"] = "train_set"
test_data["group"] = "test_set"
total = pd.concat([train_data, test_data])

categorical_vars = [
    "Elderly", "Fungal infection", "Occupation", "Gender", "Direct hospital transfer",
    "Admission route", "Restricted antimicrobial use", "Non-restricted antimicrobial use",
    "Special-class antimicrobial use", "Antimicrobial use", "Combination antimicrobial therapy",
    "Surgery", "Mechanical ventilation", "Urinary catheterization",
    "Central venous catheter (CVC)", "Disease type", "Bacterial infection",
    "ICU admission", "Isolation order", "Fever status"
]

all_vars = total.columns.values[0:len(total.columns)-1].tolist()

varbalance_table = TableOne(
    data=total, 
    columns=all_vars, 
    categorical=categorical_vars, 
    groupby="group", 
    pval=True
)

varbalance_table
varbalance_table.to_csv("Table1 varbalance_table26418.csv")

#################################### 特征工程 ##################################
###############################################################################
################ 1. 连续型变量标准化，后续加快机器学习模型收敛 #################

# 删除group变量
train_data = train_data.drop(columns='group')
continuous_vars = ['WBC', 'CRP', 'IL6', 'PCT']

# 对训练集中的连续变量进行标准化
train_data[continuous_vars] = StandardScaler().fit_transform(train_data[continuous_vars])
train_data.to_csv("train_data_scaled.csv", index=False)

# 测试集标准化
test_data = test_data.drop(columns='group')
test_data[continuous_vars] = StandardScaler().fit_transform(test_data[continuous_vars])
test_data.to_csv("test_data_scaled.csv", index=False)

##### 特征变量筛选
# 提取5000数据尝试SVM
df = pd.read_csv("train_data_scaled.csv")
print(df.head())
print(df.info())

sample_df = df.sample(n=5000, random_state=42)
sample_df.to_csv("sample_df.csv", index=False)

df = pd.read_csv("sample_df.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 2. 分离特征和目标变量
X = df.drop(columns=['Fungal infection'])
y = df['Fungal infection']

# 3. 识别分类变量和数值变量
categorical_features = ['Occupation', 'Disease type', 'Admission route']
numeric_features = [col for col in X.columns if col not in categorical_features]

# 4. 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 创建完整的SVM-RFE管道
svc = SVC(kernel="linear", probability=True)
rfecv = RFECV(
    estimator=svc,
    step=1,
    cv=5,
    scoring='accuracy',
    min_features_to_select=5
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', rfecv)
])

# 6. 执行特征选择
pipeline.fit(X, y)

# 7. 获取特征名称
preprocessor.fit(X)
transformed_feature_names = []

transformed_feature_names.extend(numeric_features)

for i, cat_col in enumerate(categorical_features):
    categories = pipeline.named_steps['preprocessor'].transformers_[1][1]\
                  .named_steps['onehot'].categories_[i][1:]
    for category in categories:
        transformed_feature_names.append(f"{cat_col}_{category}")

# 8. 创建特征重要性DataFrame
feature_ranking = pd.DataFrame({
    'Feature': transformed_feature_names,
    'Ranking': rfecv.ranking_,
    'Support': rfecv.support_
})

feature_ranking = feature_ranking.sort_values('Ranking')

# 9. 绘制特征重要性图
plt.figure(figsize=(12, 16))
plt.barh(
    feature_ranking['Feature'], 
    feature_ranking['Ranking'].max() - feature_ranking['Ranking'] + 1,
    color='skyblue'
)
plt.xlabel('Feature Importance (Higher is better)')
plt.title('Feature Importance Ranking from SVM-RFE')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('svm_rfe_feature_importance.png', dpi=300)
plt.show()

# 10. 输出选择的特征
selected_features = feature_ranking[feature_ranking['Support']]['Feature']
print("="*50)
print(f"最优特征数量: {rfecv.n_features_}")
print("="*50)
print("选择的特征变量:")
print(selected_features.to_string(index=False))
print("="*50)
print("特征排名详情:")
print(feature_ranking.to_string(index=False))

# 11. 保存结果到Excel
with pd.ExcelWriter('svm_rfe_results.xlsx') as writer:
    selected_features.to_excel(writer, sheet_name='Selected Features', index=False)
    feature_ranking.to_excel(writer, sheet_name='All Features Ranking', index=False)

# 50w数据SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import gc

chunk_size = 10000
chunks = pd.read_csv('train_data_scaled.csv', chunksize=chunk_size)

print("Fitting preprocessor...")
sample_chunk = next(chunks)

X_sample = sample_chunk.drop(columns=['Fungal infection'])
y_sample = sample_chunk['Fungal infection']

categorical_features = ['Occupation', 'Disease type', 'Admission route']
numeric_features = [col for col in X_sample.columns if col not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    sparse_threshold=1.0
)

preprocessor.fit(X_sample)

print("Processing full dataset in chunks...")
processed_chunks = []
y_full = []

chunks = pd.read_csv('数据/train_data_scaled.csv', chunksize=chunk_size)

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}...")
    X_chunk = chunk.drop(columns=['Fungal infection'])
    y_chunk = chunk['Fungal infection']
    
    X_processed = preprocessor.transform(X_chunk)
    processed_chunks.append(X_processed)
    y_full.append(y_chunk.values)
    
    del X_chunk, chunk
    gc.collect()

from scipy import sparse
X_processed = np.vstack(processed_chunks) if isinstance(processed_chunks[0], np.ndarray) else \
              sparse.vstack(processed_chunks, format='csr')
y_full = np.concatenate(y_full)

del processed_chunks
gc.collect()

print("Starting feature selection...")
svc = LinearSVC(dual=False, max_iter=1000, random_state=42, penalty='l1', C=0.1)

rfe = RFE(
    estimator=svc,
    n_features_to_select=20,
    step=0.1,
    verbose=1
)

if X_processed.shape[0] > 50000:
    print("Subsampling data for feature selection...")
    X_sub, _, y_sub, _ = train_test_split(
        X_processed, y_full, 
        train_size=50000, 
        stratify=y_full, 
        random_state=42
    )
else:
    X_sub, y_sub = X_processed, y_full

rfe.fit(X_sub, y_sub)

print("Extracting feature names...")
transformed_feature_names = []

transformed_feature_names.extend(numeric_features)

cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
for i, col in enumerate(categorical_features):
    categories = cat_encoder.categories_[i]
    if cat_encoder.drop_idx_ is not None and cat_encoder.drop_idx_[i] is not None:
        drop_idx = cat_encoder.drop_idx_[i]
        categories = [cat for j, cat in enumerate(categories) if j != drop_idx]
    for cat in categories:
        transformed_feature_names.append(f"{col}_{cat}")

feature_ranking = pd.DataFrame({
    'Feature': transformed_feature_names,
    'Ranking': rfe.ranking_,
    'Support': rfe.support_
})

feature_ranking = feature_ranking.sort_values('Ranking')

plt.figure(figsize=(12, 10))
top_features = feature_ranking.head(20)
plt.barh(
    top_features['Feature'], 
    top_features['Ranking'].max() - top_features['Ranking'] + 1,
    color='skyblue'
)
plt.xlabel('Feature Importance (Higher is better)')
plt.title('Top 20 Feature Importance Ranking from SVM-RFE')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('绘制的图片/svm_rfe_feature_importance83.png', dpi=600)
plt.show()

selected_features = feature_ranking[feature_ranking['Support']]['Feature']
print("="*50)
print(f"选择的特征数量: {rfe.n_features_}")
print("="*50)
print("选择的特征变量:")
print(selected_features.to_string(index=False))
print("="*50)
print("特征排名详情 (前30):")
print(feature_ranking.head(30).to_string(index=False))

feature_ranking.to_csv('生成的表格/svm_rfe_feature_ranking83.csv', index=False)
selected_features.to_csv('生成的表格/svm_rfe_selected_features83.csv', index=False)

joblib.dump({
    'preprocessor': preprocessor,
    'rfe': rfe,
    'feature_names': transformed_feature_names
}, 'svm_rfe_model.pkl')

# RF特征变量筛选
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns

df = pd.read_csv("数据/train_data_notscaled.csv")
print(df.info())

X = df.drop(columns=['Fungal infection'])
y = df['Fungal infection']

categorical_features = ['Occupation', 'Disease type', 'Admission route']
numeric_features = [col for col in X.columns if col not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

print("Training Random Forest model...")
pipeline.fit(X, y)

feature_names = numeric_features.copy()

onehot_columns = pipeline.named_steps['preprocessor'].transformers_[1][1]\
                   .named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(onehot_columns)

importances = pipeline.named_steps['classifier'].feature_importances_
std = np.std([tree.feature_importances_ for tree in pipeline.named_steps['classifier'].estimators_], axis=0)

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
    'Std': std
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 10))
top_features = feature_importance_df.head(22)
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('绘制的图片/rf_feature_importance83.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 18))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('All Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('绘制的图片/rf_all_feature_importance.png', dpi=300)
plt.show()

print("="*80)
print("Top 22 Features by Importance:")
print(top_features[['Feature', 'Importance']].to_string(index=False))
print("="*80)

with pd.ExcelWriter('生成的表格/rf_feature_importance_results83.xlsx') as writer:
    feature_importance_df.to_excel(writer, sheet_name='All Features', index=False)
    top_features.to_excel(writer, sheet_name='Top 22 Features', index=False)

joblib.dump(pipeline, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")

selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature']
print("\nSelected Features (Importance > 0.01):")
print(selected_features.to_string(index=False))

# Lasso回归特征变量筛选（见R语言）

# SVM-RFE、Lasso和RF特征变量求交集并绘制韦恩图
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

lasso_features = {
    'Occupation', 'Gender', 'Mechanical ventilation', 'Urinary catheterization',
    'ICU admission', 'Mechanical ventilation (days)', 'Central venous catheter (CVC)',
    'Disease type', 'Bacterial infection', 'Restricted antimicrobial use',
    'Hospital stay (days)', 'Non-restricted antimicrobial use',
    'Special-class antimicrobial use', 'Antimicrobial use', 'Duration of isolation',
    'Combination antimicrobial therapy', 'Isolation order', 'Fever status',
    'Elderly', 'WBC', 'CRP', 'PCT'
}

svm_features = {
    'Urinary catheterization', 'Mechanical ventilation',
    'Restricted antimicrobial use', 'Central venous catheter (CVC)',
    'Combination antimicrobial therapy', 'Antimicrobial use',
    'Special-class antimicrobial use', 'Bacterial infection',
    'Fever status', 'Disease type'
}

rf_features = {
    'Hospital stay (days)', 'Antimicrobial use', 'WBC', 'Restricted antimicrobial use',
    'Elderly', 'Mechanical ventilation (days)', 'Combination antimicrobial therapy',
    'PCT', 'Bacterial infection', 'Urinary catheterization (days)', 'Fever status',
    'IL6', 'Special-class antimicrobial use', 'Central venous catheter (CVC) (days)',
    'Non-restricted antimicrobial use', 'CRP', 'Urinary catheterization',
    'Central venous catheter (CVC)', 'Surgery', 'Gender',
    'Mechanical ventilation', 'Disease type'
}

common_features = lasso_features & svm_features & rf_features
print("交集特征:", common_features)

plt.figure(figsize=(10, 8))
venn = venn3(
    [lasso_features, svm_features, rf_features], 
    set_labels=('Lasso Features', 'SVM Features', 'Random Forest Features'),
    set_colors=('#FF9999', '#66B2FF', '#99FF99'), 
    alpha=0.7
)

for text in venn.set_labels:
    if text: text.set_fontsize(14)
for text in venn.subset_labels:
    if text: text.set_fontsize(12)

plt.title("Feature Selection Comparison", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('绘制的图片/韦恩图862.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 8))
venn = venn3(
    subsets=(lasso_features, svm_features, rf_features),
    set_labels=('LASSO', 'SVM', 'RF'),
    set_colors=('#1f77b4', '#ff7f0e', '#2ca02c'),
    alpha=0.7
)

for text in venn.set_labels:
    text.set_fontsize(14)
for text in venn.subset_labels:
    if text: text.set_fontsize(12)
    
plt.title("Feature Selection Methods Comparison", fontsize=16, pad=20)

if common_features:
    plt.text(
        -1.8, -1.2, 
        f"Common Features: {', '.join(common_features)}", 
        fontsize=12, 
        bbox=dict(facecolor='white', alpha=0.5)
    )

plt.tight_layout()
plt.savefig('绘制的图片/韦恩图86.png', dpi=600)
plt.show()

common_features = lasso_features & svm_features & rf_features

print("="*50)
print("特征选择结果：")
print(f"LASSO选择的特征: {', '.join(lasso_features)}")
print(f"SVM选择的特征: {', '.join(svm_features)}")
print(f"RF选择的特征: {', '.join(rf_features)}")
print("-"*50)
print(f"交集特征({len(common_features)}个): {', '.join(common_features)}")
print("="*50)

plt.figure(figsize=(10, 8))
venn = venn3(
    subsets=(lasso_features, svm_features, rf_features),
    set_labels=('LASSO', 'SVM', 'RF'),
    set_colors=('#1f77b4', '#ff7f0e', '#2ca02c'),
    alpha=0.7
)

plt.title("Feature Selection Methods Comparison", fontsize=16, pad=20, fontweight='bold')
plt.suptitle("LASSO vs SVM vs RF", fontsize=12, y=0.92, color='gray')

if common_features:
    plt.text(
        0.5, -0.15, 
        f"共同特征: {', '.join(common_features)}", 
        fontsize=12, 
        ha='center', 
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

plt.tight_layout()
plt.savefig('绘制的图片/feature_selection_venn832.png', dpi=600, bbox_inches='tight')
plt.show()

# 最终得到特征变量
###################################################################################################################
########### 基于训练集构建预测模型并基于验证集调优模型超参数（测试集数据只参与外部验证） #############
###################################################################################################################

############################# 1. Logistic模型 ##############################
train_data = pd.read_csv("train_data_notscaled.csv")
train_data['Disease type'] = train_data['Disease type'].astype('category')
print(train_data.info())

significant_vars_lasso = [
    "WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy",
    "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation",
    "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection",
    "Restricted antimicrobial use", "Special-class antimicrobial use"
]

X_train = train_data[significant_vars_lasso]
X_train_const = sm.add_constant(X_train)
y_train = train_data['Fungal infection']

logist_model = sm.Logit(y_train, X_train_const).fit(disp=0)
logist_model.summary()

y_train_pred_prob_logist = logist_model.predict(X_train_const)
auc_logist = roc_auc_score(y_train, y_train_pred_prob_logist)
auc_logist

with open("logistic_model_Lo.pkl", 'wb') as f:
    pickle.dump(logist_model, f)

############################### 2. 决策树模型 ###############################
train_data_scaled = pd.read_csv("train_data_notscaled.csv")
train_data_scaled['Disease type'] = train_data_scaled['Disease type'].astype('category')
print(train_data_scaled.info())

X = train_data_scaled[[
    "WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy",
    "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation",
    "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection",
    "Restricted antimicrobial use", "Special-class antimicrobial use"
]]
y = train_data_scaled['Fungal infection']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)

tree_default = DecisionTreeClassifier(random_state=123)
tree_default.fit(X_train, y_train)

print("模型默认参数:", pd.DataFrame.from_dict(tree_default.get_params(), orient='index'))

y_val_pred_prob_treed = tree_default.predict_proba(X_val)[:, 1]
auc_treed = roc_auc_score(y_val, y_val_pred_prob_treed)
print("默认参数模型的验证集 AUC:", auc_treed)

param_grid = {
    'max_depth': [3, 5, None],
    'min_samples_split': [60, 80, 100],
    'max_features': ['sqrt', None],
    'ccp_alpha': [0.0, 0.01, 0.1]
}

best_auc_tree = 0
tree_model_best = None
best_max_depth = None
best_min_samples_split = None
best_max_features = None
best_ccp_alpha = None

for max_depth in param_grid['max_depth']:
    for min_samples_split in param_grid['min_samples_split']:
        for max_features in param_grid['max_features']:
            for ccp_alpha in param_grid['ccp_alpha']:
                tree_model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    ccp_alpha=ccp_alpha,
                    random_state=123
                )
                tree_model.fit(X_train, y_train)
                y_val_pred_prob = tree_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_prob)

                if auc > best_auc_tree:
                    tree_model_best = tree_model
                    best_auc_tree = auc
                    best_max_depth = max_depth
                    best_min_samples_split = min_samples_split
                    best_max_features = max_features
                    best_ccp_alpha = ccp_alpha

print("最佳参数组合: max_depth =", best_max_depth,
      ", min_samples_split =", best_min_samples_split,
      ", max_features =", best_max_features,
      ", ccp_alpha =", best_ccp_alpha)
print("默认参数决策树模型的验证集 AUC:", auc_treed)
print("参数调优决策树模型的验证集 AUC:", best_auc_tree)
print("调优模型参数:", pd.DataFrame.from_dict(tree_model_best.get_params(), orient='index'))

plt.figure(figsize=(10, 5))
plot_tree(tree_model_best, feature_names=X.columns, class_names=['NO-FI', 'FI'], filled=True)
plt.savefig("tree_structure26418.jpg", dpi=500)
plt.show()

with open("tree_model.pkl", 'wb') as f:
    pickle.dump(tree_model_best, f)

################### 3. 随机森林(RF)模型 ##########################
rf_model_default = RandomForestClassifier(random_state=123, oob_score=True)
rf_model_default.fit(X_train, y_train)

print("模型默认参数:", pd.DataFrame.from_dict(rf_model_default.get_params(), orient='index'))

y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("默认参数模型的验证集 AUC:", auc_rfd)

param_grid = {
    'n_estimators': np.arange(50, 250, 50),
    'max_features': list(range(2, round(np.sqrt(X.shape[1])) + 1))
}

best_auc_rf = 0
rf_model_best = None
best_params_rf = {}

for n_estimators in param_grid['n_estimators']:
    for max_features in param_grid['max_features']:
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=123,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_val_pred_prob = rf_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_prob)

        if auc > best_auc_rf:
            best_auc_rf = auc
            rf_model_best = rf_model
            best_params_rf = {
                'n_estimators': n_estimators,
                'max_features': max_features
            }

best_ntree = best_params_rf['n_estimators']
best_mtry = best_params_rf['max_features']
print("最佳参数组合: n_estimators =", best_ntree, ", max_features =", best_mtry)
print("最佳RF参数组合:", best_params_rf)
print("默认参数RF模型的验证集 AUC:", auc_rfd)
print("参数调优RF模型的验证集 AUC:", best_auc_rf)
print("调优RF模型参数:", pd.DataFrame.from_dict(rf_model_best.get_params(), orient='index'))

with open("rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)

###################### 4. XGBoost模型 ##########################
xgb_default = XGBClassifier(random_state=123, use_label_encoder=False, eval_metric='logloss', enable_categorical='True')
xgb_default.fit(X_train, y_train)

y_val_pred_prob_xgbd = xgb_default.predict_proba(X_val)[:, 1]
auc_xgbd = roc_auc_score(y_val, y_val_pred_prob_xgbd)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0]
}

best_auc_xgb = 0
xgb_model_best = None
best_params_xgb = {}

for learning_rate in param_grid['learning_rate']:
    for max_depth in param_grid['max_depth']:
        for n_estimators in param_grid['n_estimators']:
            for subsample in param_grid['subsample']:
                xgb_model = XGBClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    random_state=123,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    enable_categorical='True'
                )
                xgb_model.fit(X_train, y_train)
                y_val_pred_prob = xgb_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_prob)

                if auc > best_auc_xgb:
                    best_auc_xgb = auc
                    xgb_model_best = xgb_model
                    best_params_xgb = {
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'subsample': subsample
                    }

print("最佳XGBoost参数组合:", best_params_xgb)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
print("参数调优XGBoost模型的验证集 AUC:", best_auc_xgb)
print("调优XGBoost模型参数:", pd.DataFrame.from_dict(xgb_model_best.get_params(), orient='index'))

with open("xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)

import joblib
joblib.dump(xgb_model_best, 'xgb_model.pkl')

###################### 5. LightGBM模型 ##########################
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)

y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)


param_grid = {
    'n_neighbors': [5, 15, 31],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}

best_auc_knn = 0
knn_model_best = None
best_params_knn = {}

# ==============================
# 1. 先用训练集子样本调参（大幅提速）
# 50万数据时，不建议每组参数都在全量训练集上跑
# ==============================
sample_size = min(100000, len(X_train))   # 可根据机器内存改成 50000

if len(X_train) > sample_size:
    X_train_tune, _, y_train_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=123
    )
else:
    X_train_tune, y_train_tune = X_train, y_train

# ==============================
# 2. 参数搜索
# 去掉重复组合：
# - euclidean == minkowski(p=2)
# - manhattan == minkowski(p=1)
# ==============================
for n_neighbors in param_grid['n_neighbors']:
    for weights in param_grid['weights']:
        for metric in param_grid['metric']:

            # 根据 metric 自动设置 p 和 algorithm
            if metric == 'minkowski':
                p = 2
                algorithm = 'kd_tree'
            elif metric == 'euclidean':
                p = 2
                algorithm = 'kd_tree'
            elif metric == 'manhattan':
                p = 1
                algorithm = 'ball_tree'

            knn_model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                p=p,
                algorithm=algorithm,
                n_jobs=-1
            )

            knn_model.fit(X_train_tune, y_train_tune)
            y_val_pred_prob = knn_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_pred_prob)

            if auc > best_auc_knn:
                best_auc_knn = auc
                best_params_knn = {
                    'n_neighbors': n_neighbors,
                    'weights': weights,
                    'metric': metric,
                    'p': p,
                    'algorithm': algorithm
                }

# ==============================
# 3. 用最优参数在全量训练集上训练最终模型
# ==============================
knn_model_best = KNeighborsClassifier(
    n_neighbors=best_params_knn['n_neighbors'],
    weights=best_params_knn['weights'],
    metric=best_params_knn['metric'],
    p=best_params_knn['p'],
    algorithm=best_params_knn['algorithm'],
    n_jobs=-1
)

knn_model_best.fit(X_train, y_train)

# ==============================
# 4. 输出保持不变
# ==============================
print("最佳KNN参数组合:", {
    'n_neighbors': best_params_knn['n_neighbors'],
    'weights': best_params_knn['weights'],
    'metric': best_params_knn['metric']
})
print("默认参数KNN模型的验证集 AUC:", auc_knnd)
print("参数调优KNN模型的验证集 AUC:", best_auc_knn)
print("调优KNN模型参数:", pd.DataFrame.from_dict(knn_model_best.get_params(), orient='index'))

# ==============================
# 5. 保存模型
# ==============================
with open("knn_model.pkl", 'wb') as f:
    pickle.dump(knn_model, f)


#####新增模型1: HistGradientBoostingClassifier 
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn
print(sklearn.__version__)          # 应 >= 0.24
from sklearn.ensemble import HistGradientBoostingClassifier
print("导入成功")


final_features = ["Antimicrobial use", "Urinary catheterization", "Fever status",
                  "Combination antimicrobial therapy", "Special-class antimicrobial use",
                  "Restricted antimicrobial use", "Bacterial infection",
                  "Central venous catheter (CVC)", "Disease type", "Mechanical ventilation"]

# 构建训练集和验证集（用于模型调优）
train_data_scaled = pd.read_csv("train_data_notscaled.csv")
train_data_scaled['Disease type'] = train_data_scaled['Disease type'].astype('category')
X = train_data_scaled[final_features]
y = train_data_scaled['Fungal infection']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
# 该模型可直接处理缺失值和分类特征，需指明分类列
categorical_columns = ["Elderly", "Combination antimicrobial therapy",
    "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation",
    "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection",
    "Restricted antimicrobial use", "Special-class antimicrobial use"]  # 根据实际数据调整
# 注意：最终特征中可能包含这些分类变量，需从X中获取分类列名
categorical_features_in_use = [col for col in final_features if col in categorical_columns]
hgb_default = HistGradientBoostingClassifier(random_state=123, categorical_features=categorical_features_in_use)
hgb_default.fit(X_train, y_train)
y_val_pred_prob_hgbd = hgb_default.predict_proba(X_val)[:, 1]
auc_hgbd = roc_auc_score(y_val, y_val_pred_prob_hgbd)
print(f"默认参数HistGradientBoosting验证集 AUC: {auc_hgbd:.4f}")

# 超参数网格搜索
param_grid_hgb = {
    'learning_rate': [0.1, 0.2],
    'max_depth':  [5, 10],
    'max_iter': [100],
    'l2_regularization': [0.0, 0.1]
}
best_auc_hgb = 0
best_hgb = None
best_params_hgb = {}
for lr in param_grid_hgb['learning_rate']:
    for md in param_grid_hgb['max_depth']:
        for mi in param_grid_hgb['max_iter']:
            for l2 in param_grid_hgb['l2_regularization']:
                model = HistGradientBoostingClassifier(
                    learning_rate=lr, max_depth=md, max_iter=mi, l2_regularization=l2,
                    random_state=123, categorical_features=categorical_features_in_use
                )
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_val)[:, 1]
                auc_val = roc_auc_score(y_val, prob)
                if auc_val > best_auc_hgb:
                    best_auc_hgb = auc_val
                    best_hgb = model
                    best_params_hgb = {'learning_rate': lr, 'max_depth': md, 'max_iter': mi, 'l2_regularization': l2}
print("最佳HistGradientBoosting参数:", best_params_hgb)
print(f"调优后验证集 AUC: {best_auc_hgb:.4f}")

# 保存模型
with open("hgb_model.pkl", 'wb') as f:
    pickle.dump(best_hgb, f)
################################################################################################
################################## 验证数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
train_data = pd.read_csv("train_data_notscaled.csv")
print(train_data.info())
train_data['Disease type'] = train_data['Disease type'].astype('category')

significant_vars_lasso = [
    "WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy",
    "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation",
    "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection",
    "Restricted antimicrobial use", "Special-class antimicrobial use"
]

X_train_logist = train_data[significant_vars_lasso]
X_train_logist_const = sm.add_constant(X_train_logist)
y_train_logist = train_data['Fungal infection']

train_data_scaled = pd.read_csv("train_data_notscaled.csv")
train_data_scaled['Disease type'] = train_data_scaled['Disease type'].astype('category')

X = train_data_scaled[[
    "WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy",
    "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation",
    "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection",
    "Restricted antimicrobial use", "Special-class antimicrobial use"
]]
y = train_data_scaled['Fungal infection']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)

###################### 1. 加载训练好的模型 ##########################
with open("logistic_model_Lo.pkl", 'rb') as f:
    logist_model = pickle.load(f)

with open("tree_model.pkl", 'rb') as f:
    tree_model = pickle.load(f)

with open("rf_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)

with open("lgb_model.pkl", 'rb') as f:
    lgb_model = pickle.load(f)

with open("knn_model.pkl", 'rb') as f:
    knn_model = pickle.load(f)

with open("hgb_model.pkl", 'rb') as f:
    hgb_model = pickle.load(f)

## run end ##

###################### 2. 得到验证数据集预测结果，包括预测概率和预测分类 ##########################
# Logistic
y_train_pred_prob_logist = logist_model.predict(X_train_logist_const)
y_train_pred_logist = (y_train_pred_prob_logist >= 0.5).astype(int)

# 决策树
y_val_pred_prob_tree = tree_model.predict_proba(X_val)[:, 1]
y_val_pred_tree = (y_val_pred_prob_tree >= 0.5).astype(int)

# 随机森林
y_val_pred_prob_rf = rf_model.predict_proba(X_val)[:, 1]
y_val_pred_rf = (y_val_pred_prob_rf >= 0.5).astype(int)

# XGBoost
y_val_pred_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred_xgb = (y_val_pred_prob_xgb >= 0.5).astype(int)

# LightGBM
y_val_pred_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
y_val_pred_lgb = (y_val_pred_prob_lgb >= 0.5).astype(int)

# KNN（新增）
y_val_pred_prob_knn = knn_model.predict_proba(X_val)[:, 1]
y_val_pred_knn = (y_val_pred_prob_knn >= 0.5).astype(int)

# hgb（新增）
y_val_pred_prob_hgb = hgb_model.predict_proba(X_val)[:, 1]
y_val_pred_hgb = (y_val_pred_prob_hgb >= 0.5).astype(int)

###################### 3. 计算混淆矩阵并可视化 ##########################
def CM_plot(cm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No FI', 'FI'], yticklabels=['No FI', 'FI'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

cm_logist = confusion_matrix(y_train_logist, y_train_pred_logist)
CM_plot(cm_logist)

cm_tree = confusion_matrix(y_val, y_val_pred_tree)
CM_plot(cm_tree)

cm_rf = confusion_matrix(y_val, y_val_pred_rf)
CM_plot(cm_rf)

cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
CM_plot(cm_xgb)

cm_lgb = confusion_matrix(y_val, y_val_pred_lgb)
CM_plot(cm_lgb)

cm_knn = confusion_matrix(y_val, y_val_pred_knn)   # 新增
CM_plot(cm_knn)

cm_hgb = confusion_matrix(y_val, y_val_pred_hgb)   # 新增
CM_plot(cm_hgb)

###################### 4. 计算准确率、精确率、灵敏度、f1分数、特异度 ##########################
def calculate_acc_pre_sen_f1_spc(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return accuracy, precision, sensitivity, f1score, specificity

accuracy_logist, precision_logist, sensitivity_logist, f1_logist, specificity_logist = calculate_acc_pre_sen_f1_spc(cm_logist)
print(f"Logistic Model → Accuracy: {accuracy_logist:.3f}, Precision: {precision_logist:.3f}, Sensitivity: {sensitivity_logist:.3f}, F1 Score: {f1_logist:.3f}, Specificity: {specificity_logist:.3f}")

accuracy_tree, precision_tree, sensitivity_tree, f1_tree, specificity_tree = calculate_acc_pre_sen_f1_spc(cm_tree)
print(f"Decision Tree Model → Accuracy: {accuracy_tree:.3f}, Precision: {precision_tree:.3f}, Sensitivity: {sensitivity_tree:.3f}, F1 Score: {f1_tree:.3f}, Specificity: {specificity_tree:.3f}")

accuracy_rf, precision_rf, sensitivity_rf, f1_rf, specificity_rf = calculate_acc_pre_sen_f1_spc(cm_rf)
print(f"Random Forest Model → Accuracy: {accuracy_rf:.3f}, Precision: {precision_rf:.3f}, Sensitivity: {sensitivity_rf:.3f}, F1 Score: {f1_rf:.3f}, Specificity: {specificity_rf:.3f}")

accuracy_xgb, precision_xgb, sensitivity_xgb, f1_xgb, specificity_xgb = calculate_acc_pre_sen_f1_spc(cm_xgb)
print(f"XGBoost Model → Accuracy: {accuracy_xgb:.3f}, Precision: {precision_xgb:.3f}, Sensitivity: {sensitivity_xgb:.3f}, F1 Score: {f1_xgb:.3f}, Specificity: {specificity_xgb:.3f}")

accuracy_lgb, precision_lgb, sensitivity_lgb, f1_lgb, specificity_lgb = calculate_acc_pre_sen_f1_spc(cm_lgb)
print(f"LightGBM Model → Accuracy: {accuracy_lgb:.3f}, Precision: {precision_lgb:.3f}, Sensitivity: {sensitivity_lgb:.3f}, F1 Score: {f1_lgb:.3f}, Specificity: {specificity_lgb:.3f}")

accuracy_knn, precision_knn, sensitivity_knn, f1_knn, specificity_knn = calculate_acc_pre_sen_f1_spc(cm_knn)
print(f"KNN Model → Accuracy: {accuracy_knn:.3f}, Precision: {precision_knn:.3f}, Sensitivity: {sensitivity_knn:.3f}, F1 Score: {f1_knn:.3f}, Specificity: {specificity_knn:.3f}")

accuracy_hgb, precision_hgb, sensitivity_hgb, f1_hgb, specificity_hgb = calculate_acc_pre_sen_f1_spc(cm_hgb)
print(f"HGB Model → Accuracy: {accuracy_hgb:.3f}, Precision: {precision_hgb:.3f}, Sensitivity: {sensitivity_hgb:.3f}, F1 Score: {f1_hgb:.3f}, Specificity: {specificity_hgb:.3f}")

###################### 5. 计算AUC及其95%置信区间 ##########################
def calculate_auc(y_label, y_pred_prob):
    auc_value = roc_auc_score(y_label, y_pred_prob)
    se_auc = np.sqrt((auc_value * (1 - auc_value)) / len(y_label))
    z = norm.ppf(0.975)
    auc_ci_lower = auc_value - z * se_auc
    auc_ci_upper = auc_value + z * se_auc
    return auc_value, auc_ci_lower, auc_ci_upper

auc_value_logist, auc_ci_lower_logist, auc_ci_upper_logist = calculate_auc(y_train_logist, y_train_pred_prob_logist)
print(f"Logistic Model AUC: {auc_value_logist:.3f} (95% CI: {auc_ci_lower_logist:.3f} - {auc_ci_upper_logist:.3f})")

auc_value_tree, auc_ci_lower_tree, auc_ci_upper_tree = calculate_auc(y_val, y_val_pred_prob_tree)
print(f"Decision Tree Model AUC: {auc_value_tree:.3f} (95% CI: {auc_ci_lower_tree:.3f} - {auc_ci_upper_tree:.3f})")

auc_value_rf, auc_ci_lower_rf, auc_ci_upper_rf = calculate_auc(y_val, y_val_pred_prob_rf)
print(f"Random Forest Model AUC: {auc_value_rf:.3f} (95% CI: {auc_ci_lower_rf:.3f} - {auc_ci_upper_rf:.3f})")

auc_value_xgb, auc_ci_lower_xgb, auc_ci_upper_xgb = calculate_auc(y_val, y_val_pred_prob_xgb)
print(f"XGBoost Model AUC: {auc_value_xgb:.3f} (95% CI: {auc_ci_lower_xgb:.3f} - {auc_ci_upper_xgb:.3f})")

auc_value_lgb, auc_ci_lower_lgb, auc_ci_upper_lgb = calculate_auc(y_val, y_val_pred_prob_lgb)
print(f"LightGBM Model AUC: {auc_value_lgb:.3f} (95% CI: {auc_ci_lower_lgb:.3f} - {auc_ci_upper_lgb:.3f})")

auc_value_knn, auc_ci_lower_knn, auc_ci_upper_knn = calculate_auc(y_val, y_val_pred_prob_knn)
print(f"KNN Model AUC: {auc_value_knn:.3f} (95% CI: {auc_ci_lower_knn:.3f} - {auc_ci_upper_knn:.3f})")

auc_value_hgb, auc_ci_lower_hgb, auc_ci_upper_hgb = calculate_auc(y_val, y_val_pred_prob_hgb)
print(f"HGB Model AUC: {auc_value_hgb:.3f} (95% CI: {auc_ci_lower_hgb:.3f} - {auc_ci_upper_hgb:.3f})")

###################### 6. 绘制ROC曲线 ##########################
def ROC_plot(y_label, y_pred_prob, auc_value):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.show()

ROC_plot(y_train_logist, y_train_pred_prob_logist, auc_value_logist)
ROC_plot(y_val, y_val_pred_prob_tree, auc_value_tree)
ROC_plot(y_val, y_val_pred_prob_rf, auc_value_rf)
ROC_plot(y_val, y_val_pred_prob_xgb, auc_value_xgb)
ROC_plot(y_val, y_val_pred_prob_lgb, auc_value_lgb)
ROC_plot(y_val, y_val_pred_prob_knn, auc_value_knn)   # 新增
ROC_plot(y_val, y_val_pred_prob_hgb, auc_value_hgb)   # 新增

###################### 7. 绘制校准曲线 ##########################
def CaliC_plot(y_label, y_pred_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_label, y_pred_prob, n_bins=n_bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()

CaliC_plot(y_train_logist, y_train_pred_prob_logist)
CaliC_plot(y_train_logist, y_train_pred_prob_logist, n_bins=8)
CaliC_plot(y_val, y_val_pred_prob_tree)
CaliC_plot(y_val, y_val_pred_prob_rf)
CaliC_plot(y_val, y_val_pred_prob_xgb)
CaliC_plot(y_val, y_val_pred_prob_lgb)
CaliC_plot(y_val, y_val_pred_prob_knn)   # 新增
CaliC_plot(y_val, y_val_pred_prob_hgb)   # 新增

###################### 8. 绘制决策分析曲线 (DCA) ##########################
def calculate_net_benefi(y_label, y_pred_prob, thresholds=np.linspace(0.01, 1, 100)):
    net_benefit_model = []
    net_benefit_alltrt = []
    net_benefits_notrt = [0] * len(thresholds)
    total_obs = len(y_label)

    for thresh in thresholds:
        y_pred_label = y_pred_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        net_benefit = (tp / total_obs) - (fp / total_obs) * (thresh / (1 - thresh))
        net_benefit_model.append(net_benefit)

        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total_right = tp + tn
        net_benefit = (tp / total_right) - (tn / total_right) * (thresh / (1 - thresh))
        net_benefit_alltrt.append(net_benefit)

    return net_benefit_model, net_benefit_alltrt, net_benefits_notrt

def DCA_plot(net_benefit_model, net_benefit_alltrt, net_benefits_notrt,
             thresholds=np.linspace(0.01, 0.99, 100)):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model, label="Model Net Benefit", color='blue', linewidth=2)
    plt.plot(thresholds, net_benefit_alltrt, label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefits_notrt, linestyle='--', color='green', label="Treat None", linewidth=2)
    plt.xlabel("Threshold Probability")
    plt.ylim(-0.10, np.nanmax(np.array(net_benefit_model)) + 0.05)
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

net_benefit_logist, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_train_logist, y_train_pred_prob_logist)
DCA_plot(net_benefit_logist, net_benefit_alltrt, net_benefits_notrt)

net_benefit_tree, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_tree)
DCA_plot(net_benefit_tree, net_benefit_alltrt, net_benefits_notrt)

net_benefit_rf, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_rf)
DCA_plot(net_benefit_rf, net_benefit_alltrt, net_benefits_notrt)

net_benefit_xgb, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_xgb)
DCA_plot(net_benefit_xgb, net_benefit_alltrt, net_benefits_notrt)

net_benefit_lgb, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_lgb)
DCA_plot(net_benefit_lgb, net_benefit_alltrt, net_benefits_notrt)

net_benefit_knn, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_knn)
DCA_plot(net_benefit_knn, net_benefit_alltrt, net_benefits_notrt)

net_benefit_hgb, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_hgb)
DCA_plot(net_benefit_hgb, net_benefit_alltrt, net_benefits_notrt)

###################### 9. 所有模型的训练集/验证集预测效果汇总 ##########################
model_results_validation = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "KNN", "SVM"],
    "AUC": [auc_value_logist, auc_value_tree, auc_value_rf, auc_value_xgb, auc_value_lgb, auc_value_knn, auc_value_hgb],
    "AUC 95% CI Lower": [auc_ci_lower_logist, auc_ci_lower_tree, auc_ci_lower_rf, auc_ci_lower_xgb, auc_ci_lower_lgb, auc_ci_lower_knn, auc_ci_lower_hgb],
    "AUC 95% CI Upper": [auc_ci_upper_logist, auc_ci_upper_tree, auc_ci_upper_rf, auc_ci_upper_xgb, auc_ci_upper_lgb, auc_ci_upper_knn, auc_ci_upper_hgb],
    "Accuracy": [accuracy_logist, accuracy_tree, accuracy_rf, accuracy_xgb, accuracy_lgb, accuracy_knn, accuracy_hgb],
    "Precision": [precision_logist, precision_tree, precision_rf, precision_xgb, precision_lgb, precision_knn, precision_hgb],
    "Sensitivity": [sensitivity_logist, sensitivity_tree, sensitivity_rf, sensitivity_xgb, sensitivity_lgb, sensitivity_knn, sensitivity_hgb],
    "Specificity": [specificity_logist, specificity_tree, specificity_rf, specificity_xgb, specificity_lgb, specificity_knn, specificity_hgb],
    "F1 Score": [f1_logist, f1_tree, f1_rf, f1_xgb, f1_lgb, f1_knn, f1_hgb]
})
model_results_validation
model_results_validation.to_csv("model_performance_training260419.csv", index=False)

plt.figure(figsize=(8, 6))
models = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist, auc_value_logist),
    "Decision Tree": (y_val, y_val_pred_prob_tree, auc_value_tree),
    "Random Forest": (y_val, y_val_pred_prob_rf, auc_value_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb, auc_value_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb, auc_value_lgb),
    "KNN": (y_val, y_val_pred_prob_knn, auc_value_knn),
    "SVM": (y_val, y_val_pred_prob_hgb, auc_value_hgb),
}
for model_name, (y_true, y_pred_prob, auc_value) in models.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("ROC_curves_allmodel_training260419.png", dpi=600)
plt.show()

plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves for All Models")
plt.legend(loc='upper left')
plt.grid()
plt.savefig("Calibration_curves_allmodel_training260419.png", dpi=600)
plt.show()

plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_true, y_pred_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), net_benefit, label=model_name)
plt.plot(np.linspace(0.01, 0.99, 100), net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), net_benefits_notrt, linestyle="--", color="green", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylim(-0.10, np.nanmax(np.array(net_benefit)) + 0.05)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("DCA_curves_allmodel_training260419.png", dpi=600)
plt.show()

################################################################################################
################################## 测试数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
test_data = pd.read_csv("test_data_notscaled.csv")
test_data['Disease type'] = test_data['Disease type'].astype('category')

significant_vars_lasso = [
    "WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy",
    "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation",
    "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection",
    "Restricted antimicrobial use", "Special-class antimicrobial use"
]

X_test_logist = test_data[significant_vars_lasso]
X_test_logist_const = sm.add_constant(X_test_logist)
y_test_logist = test_data['Fungal infection']

test_data_scaled = pd.read_csv("test_data_notscaled.csv")
test_data_scaled['Disease type'] = test_data_scaled['Disease type'].astype('category')

X_test = test_data_scaled[[
    "WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy",
    "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation",
    "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection",
    "Restricted antimicrobial use", "Special-class antimicrobial use"
]]
y_test = test_data_scaled['Fungal infection']
## run end ##

###################### 1. 计算测试数据集预测结果 ##########################
y_test_pred_prob_logist = logist_model.predict(X_test_logist_const)
y_test_pred_logist = (y_test_pred_prob_logist >= 0.5).astype(int)

y_test_pred_prob_tree = tree_model.predict_proba(X_test)[:, 1]
y_test_pred_tree = (y_test_pred_prob_tree >= 0.5).astype(int)

y_test_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_test_pred_rf = (y_test_pred_prob_rf >= 0.5).astype(int)

y_test_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred_xgb = (y_test_pred_prob_xgb >= 0.5).astype(int)

y_test_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_test_pred_lgb = (y_test_pred_prob_lgb >= 0.5).astype(int)

# KNN（新增）
y_test_pred_prob_knn = knn_model.predict_proba(X_test)[:, 1]
y_test_pred_knn = (y_test_pred_prob_knn >= 0.5).astype(int)

# SVM（新增）
y_test_pred_prob_hgb = hgb_model.predict_proba(X_test)[:, 1]
y_test_pred_hgb = (y_test_pred_prob_hgb >= 0.5).astype(int)

###################### 2. 计算混淆矩阵并可视化 ##########################
cm_logist_test = confusion_matrix(y_test_logist, y_test_pred_logist)
CM_plot(cm_logist_test)

cm_tree_test = confusion_matrix(y_test, y_test_pred_tree)
CM_plot(cm_tree_test)

cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)
CM_plot(cm_rf_test)

cm_xgb_test = confusion_matrix(y_test, y_test_pred_xgb)
CM_plot(cm_xgb_test)

cm_lgb_test = confusion_matrix(y_test, y_test_pred_lgb)
CM_plot(cm_lgb_test)

cm_knn_test = confusion_matrix(y_test, y_test_pred_knn)
CM_plot(cm_knn_test)

cm_hgb_test = confusion_matrix(y_test, y_test_pred_hgb)
CM_plot(cm_hgb_test)

###################### 3. 计算准确率、精确率、灵敏度、f1分数、特异度 ##########################
accuracy_logist_test, precision_logist_test, sensitivity_logist_test, f1_logist_test, specificity_logist_test = calculate_acc_pre_sen_f1_spc(cm_logist_test)
accuracy_tree_test, precision_tree_test, sensitivity_tree_test, f1_tree_test, specificity_tree_test = calculate_acc_pre_sen_f1_spc(cm_tree_test)
accuracy_rf_test, precision_rf_test, sensitivity_rf_test, f1_rf_test, specificity_rf_test = calculate_acc_pre_sen_f1_spc(cm_rf_test)
accuracy_xgb_test, precision_xgb_test, sensitivity_xgb_test, f1_xgb_test, specificity_xgb_test = calculate_acc_pre_sen_f1_spc(cm_xgb_test)
accuracy_lgb_test, precision_lgb_test, sensitivity_lgb_test, f1_lgb_test, specificity_lgb_test = calculate_acc_pre_sen_f1_spc(cm_lgb_test)
accuracy_knn_test, precision_knn_test, sensitivity_knn_test, f1_knn_test, specificity_knn_test = calculate_acc_pre_sen_f1_spc(cm_knn_test)
accuracy_hgb_test, precision_hgb_test, sensitivity_hgb_test, f1_hgb_test, specificity_hgb_test = calculate_acc_pre_sen_f1_spc(cm_hgb_test)

###################### 4. 计算AUC及其95%置信区间 ##########################
auc_value_logist_test, auc_ci_lower_logist_test, auc_ci_upper_logist_test = calculate_auc(y_test_logist, y_test_pred_prob_logist)
auc_value_tree_test, auc_ci_lower_tree_test, auc_ci_upper_tree_test = calculate_auc(y_test, y_test_pred_prob_tree)
auc_value_rf_test, auc_ci_lower_rf_test, auc_ci_upper_rf_test = calculate_auc(y_test, y_test_pred_prob_rf)
auc_value_xgb_test, auc_ci_lower_xgb_test, auc_ci_upper_xgb_test = calculate_auc(y_test, y_test_pred_prob_xgb)
auc_value_lgb_test, auc_ci_lower_lgb_test, auc_ci_upper_lgb_test = calculate_auc(y_test, y_test_pred_prob_lgb)
auc_value_knn_test, auc_ci_lower_knn_test, auc_ci_upper_knn_test = calculate_auc(y_test, y_test_pred_prob_knn)
auc_value_hgb_test, auc_ci_lower_hgb_test, auc_ci_upper_hgb_test = calculate_auc(y_test, y_test_pred_prob_hgb)

###################### 5. 所有模型的测试集预测效果汇总 ##########################
model_results_test = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "KNN", "SVM"],
    "AUC": [auc_value_logist_test, auc_value_tree_test, auc_value_rf_test, auc_value_xgb_test, auc_value_lgb_test, auc_value_knn_test, auc_value_hgb_test],
    "AUC 95% CI Lower": [auc_ci_lower_logist_test, auc_ci_lower_tree_test, auc_ci_lower_rf_test, auc_ci_lower_xgb_test, auc_ci_lower_lgb_test, auc_ci_lower_knn_test, auc_ci_lower_hgb_test],
    "AUC 95% CI Upper": [auc_ci_upper_logist_test, auc_ci_upper_tree_test, auc_ci_upper_rf_test, auc_ci_upper_xgb_test, auc_ci_upper_lgb_test, auc_ci_upper_knn_test, auc_ci_upper_hgb_test],
    "Accuracy": [accuracy_logist_test, accuracy_tree_test, accuracy_rf_test, accuracy_xgb_test, accuracy_lgb_test, accuracy_knn_test, accuracy_hgb_test],
    "Precision": [precision_logist_test, precision_tree_test, precision_rf_test, precision_xgb_test, precision_lgb_test, precision_knn_test, precision_hgb_test],
    "Sensitivity": [sensitivity_logist_test, sensitivity_tree_test, sensitivity_rf_test, sensitivity_xgb_test, sensitivity_lgb_test, sensitivity_knn_test, sensitivity_hgb_test],
    "Specificity": [specificity_logist_test, specificity_tree_test, specificity_rf_test, specificity_xgb_test, specificity_lgb_test, specificity_knn_test, specificity_hgb_test],
    "F1 Score": [f1_logist_test, f1_tree_test, f1_rf_test, f1_xgb_test, f1_lgb_test, f1_knn_test, f1_hgb_test]
})
model_results_test
model_results_test.to_csv("model_performance_test260419.csv", index=False)

###################### 6. 绘制ROC曲线 ##########################
plt.figure(figsize=(8, 6))
models_test = {
    "Logistic": (y_test_logist, y_test_pred_prob_logist, auc_value_logist_test),
    "Decision Tree": (y_test, y_test_pred_prob_tree, auc_value_tree_test),
    "Random Forest": (y_test, y_test_pred_prob_rf, auc_value_rf_test),
    "XGBoost": (y_test, y_test_pred_prob_xgb, auc_value_xgb_test),
    "LightGBM": (y_test, y_test_pred_prob_lgb, auc_value_lgb_test),
    "KNN": (y_test, y_test_pred_prob_knn, auc_value_knn_test),
    "SVM": (y_test, y_test_pred_prob_hgb, auc_value_hgb_test),
}
for model_name, (y_true, y_pred_prob, auc_value) in models_test.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models (Test Set)")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("ROC_curves_allmodel_test260419.png", dpi=600)
plt.show()

###################### 7. 绘制校准曲线 ##########################
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models_test.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves for All Models (Test Set)")
plt.legend(loc='upper left')
plt.grid()
plt.savefig("Calibration_curves_allmodel_test260419.png", dpi=600)
plt.show()

###################### 8. 绘制DCA曲线 ##########################
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models_test.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_true, y_pred_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), net_benefit, label=model_name)
plt.plot(np.linspace(0.01, 0.99, 100), net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), net_benefits_notrt, linestyle="--", color="green", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylim(-0.10, np.nanmax(np.array(net_benefit)) + 0.05)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models (Test Set)")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("DCA_curves_allmodel_test260419.png", dpi=600)
plt.show()