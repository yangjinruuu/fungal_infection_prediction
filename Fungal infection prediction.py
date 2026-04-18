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

os.getcwd() # 当前工作路径
data = pd.read_csv("数据/分析数据83英文.csv")

print(data.head())  # 查看前五行数据
print(data.info()) 

data['Occupation'] = data['Occupation'].astype('category')
data['Disease type'] = data['Disease type'].astype('category')
data['Admission route'] = data['Admission route'].astype('category')
###############################################################################
#################################### 数据拆分 ########################################
#####################################################################################
##################### 1. 数据集拆分：训练集和测试集 ######################

train_data, test_data = train_test_split(data, test_size=0.3, # 拆分数据
                                         stratify=data["Fungal infection"], random_state=2025) # https://scikit-learn.org.cn/view/649.html

# 保存这两个数据集
train_data.to_csv("train_data_notscaled.csv", index=False) # 保存训练集
test_data.to_csv("数据/test_data_notscaled.csv", index=False) # 保存测试集

######################## 2. 查看训练集vs测试集的变量均衡性 ########################
# 为两个数据集都生成一个group变量，便于后续变量均衡性检查
train_data["group"] = "train_set"
test_data["group"] = "test_set"
# 合并这两个数据集
total = pd.concat([train_data, test_data]) # 默认按行合并
# 创建描述性统计表
categorical_vars = ["Elderly", "Fungal infection", "Occupation", "Gender", "Direct hospital transfer", "Admission route",  "Restricted antimicrobial use", "Non-restricted antimicrobial use", "Special-class antimicrobial use", "Antimicrobial use", "Combination antimicrobial therapy", "Surgery","Mechanical ventilation", "Urinary catheterization", "Central venous catheter (CVC)", "Disease type", "Bacterial infection", "ICU admission", "Isolation order", "Fever status"] # 分类变量的变量名
all_vars = total.columns.values[0:len(total.columns)-1].tolist() # 除了'group'变量外的所有变量的变量名
varbalance_table = TableOne(data=total, columns=all_vars, 
                            categorical=categorical_vars, groupby="group", pval=True) # 以group为分组，创建描述性统计表
# 查看变量均衡情况
varbalance_table
# 保存为csv文件
varbalance_table.to_csv("生成的表格/Table1 varbalance_table83.csv")

#################################### 特征工程 ########################################
#####################################################################################
################ 1. 连续型变量标准化，后续加快机器学习模型收敛 #################
## 首先训练集连续变量的标准化
#删除group变量
train_data = train_data.drop(columns='group')
train_data
continuous_vars = ['WBC', 'CRP', 'IL6', 'PCT'] # 连续变量的变量名
train_data[continuous_vars] = StandardScaler().fit_transform(train_data[continuous_vars]) # 对训练集中的连续变量进行标准化
train_data
# 保存标准化后的训练集
train_data.to_csv("数据/train_data_scaled.csv", index=False)

## 接下来测试集连续变量的标准化
#删除group变量
test_data = test_data.drop(columns='group')
test_data[continuous_vars] = StandardScaler().fit_transform(test_data[continuous_vars])
# 保存标准化后的测试集
test_data.to_csv("数据/test_data_scaled.csv", index=False)




#####特征变量筛选
#提取5000数据尝试SVM
#SANMPLE数据-svm-ref特征变量筛选（需要对数据进行标准化），无序多分类变量按哑变量处理
df = pd.read_csv("数据/train_data_scaled.csv")
print(df.head())  # 查看前五行数据
print(df.info())  # 查看数据结构
sample_df = df.sample(n=5000, random_state=42)
sample_df.to_csv("sample_df.csv", index=False)
df= pd.read_csv("sample_df.csv")
df = df.drop(columns='Unnamed: 0')

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

# 7. 获取特征名称（处理独热编码后的特征名）
preprocessor.fit(X)
transformed_feature_names = []

# 获取数值特征名
transformed_feature_names.extend(numeric_features)

# 获取分类特征名（独热编码后）
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

# 按排名排序（排名1表示最重要的特征）
feature_ranking = feature_ranking.sort_values('Ranking')

# 9. 绘制特征重要性图
plt.figure(figsize=(12, 16))
plt.barh(feature_ranking['Feature'], 
         feature_ranking['Ranking'].max() - feature_ranking['Ranking'] + 1,
         color='skyblue')
plt.xlabel('Feature Importance (Higher is better)')
plt.title('Feature Importance Ranking from SVM-RFE')
plt.gca().invert_yaxis()  # 最重要的特征显示在顶部
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


#SVM-REF方法
#50w数据SVM
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

# 1. 分块读取大型Excel文件
chunk_size = 10000  # 每次处理1万行
chunks = pd.read_csv('数据/train_data_scaled.csv',  chunksize=chunk_size)

# 2. 初始化预处理转换器（先拟合转换器）
print("Fitting preprocessor...")
sample_chunk = next(chunks)  # 取第一个chunk用于拟合

# 分离特征和目标变量
X_sample = sample_chunk.drop(columns=['Fungal infection'])
y_sample = sample_chunk['Fungal infection']

# 识别分类变量和数值变量
categorical_features = ['Occupation', 'Disease type', 'Admission route']
numeric_features = [col for col in X_sample.columns if col not in categorical_features]

# 创建预处理管道
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
    sparse_threshold=1.0  # 强制返回稀疏矩阵
)

# 在样本上拟合预处理器
preprocessor.fit(X_sample)

# 3. 处理完整数据集（分块处理）
print("Processing full dataset in chunks...")
processed_chunks = []
y_full = []

# 重置chunks迭代器
chunks = pd.read_csv('数据/train_data_scaled.csv',  chunksize=chunk_size)

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}...")
    X_chunk = chunk.drop(columns=['Fungal infection'])
    y_chunk = chunk['Fungal infection']
    
    # 转换当前chunk
    X_processed = preprocessor.transform(X_chunk)
    processed_chunks.append(X_processed)
    y_full.append(y_chunk.values)
    
    # 释放内存
    del X_chunk, chunk
    gc.collect()

# 合并处理后的数据
from scipy import sparse
X_processed = np.vstack(processed_chunks) if isinstance(processed_chunks[0], np.ndarray) else \
              sparse.vstack(processed_chunks, format='csr')
y_full = np.concatenate(y_full)

# 释放内存
del processed_chunks
gc.collect()

# 4. 特征选择（使用线性SVM和RFE）
print("Starting feature selection...")
svc = LinearSVC(dual=False, max_iter=1000, random_state=42, penalty='l1', C=0.1)  # 使用L1正则化

# 使用RFE代替RFECV以提高效率
rfe = RFE(
    estimator=svc,
    n_features_to_select=20,  # 直接指定要选择的特征数量
    step=0.1,  # 每次迭代移除10%的特征
    verbose=1
)

# 在数据子集上训练（加速计算）
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

# 执行特征选择
rfe.fit(X_sub, y_sub)

# 5. 获取特征名称
print("Extracting feature names...")
transformed_feature_names = []

# 获取数值特征名
transformed_feature_names.extend(numeric_features)

# 获取分类特征名（独热编码后）
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
for i, col in enumerate(categorical_features):
    categories = cat_encoder.categories_[i]
    # 只添加非基准类别
    if cat_encoder.drop_idx_ is not None and cat_encoder.drop_idx_[i] is not None:
        drop_idx = cat_encoder.drop_idx_[i]
        categories = [cat for j, cat in enumerate(categories) if j != drop_idx]
    for cat in categories:
        transformed_feature_names.append(f"{col}_{cat}")

# 6. 创建特征重要性DataFrame
feature_ranking = pd.DataFrame({
    'Feature': transformed_feature_names,
    'Ranking': rfe.ranking_,
    'Support': rfe.support_
})

# 按排名排序（排名1表示最重要的特征）
feature_ranking = feature_ranking.sort_values('Ranking')

# 7. 绘制特征重要性图（只展示前20个特征）
plt.figure(figsize=(12, 10))
top_features = feature_ranking.head(20)
plt.barh(top_features['Feature'], 
         top_features['Ranking'].max() - top_features['Ranking'] + 1,
         color='skyblue')
plt.xlabel('Feature Importance (Higher is better)')
plt.title('Top 20 Feature Importance Ranking from SVM-RFE')
plt.gca().invert_yaxis()  # 最重要的特征显示在顶部
plt.tight_layout()
plt.savefig('绘制的图片/svm_rfe_feature_importance83.png', dpi=600)
plt.show()

# 8. 输出选择的特征
selected_features = feature_ranking[feature_ranking['Support']]['Feature']
print("="*50)
print(f"选择的特征数量: {rfe.n_features_}")
print("="*50)
print("选择的特征变量:")
print(selected_features.to_string(index=False))
print("="*50)
print("特征排名详情 (前30):")
print(feature_ranking.head(30).to_string(index=False))

# 9. 保存结果到CSV（避免Excel内存问题）
feature_ranking.to_csv('生成的表格/svm_rfe_feature_ranking83.csv', index=False)
selected_features.to_csv('生成的表格/svm_rfe_selected_features83.csv', index=False)

# 10. 保存模型和预处理管道
joblib.dump({
    'preprocessor': preprocessor,
    'rfe': rfe,
    'feature_names': transformed_feature_names
}, 'svm_rfe_model.pkl')



#RF特征变量筛选

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

# 2. 分离特征和目标变量
X = df.drop(columns=['Fungal infection'])
y = df['Fungal infection']

# 3. 识别分类变量和数值变量
categorical_features = ['Occupation', 'Disease type', 'Admission route']
numeric_features = [col for col in X.columns if col not in categorical_features]

# 4. 创建预处理管道
# 数值特征管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

# 分类特征管道
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 创建完整的随机森林管道
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # 使用所有CPU核心
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

# 6. 训练模型
print("Training Random Forest model...")
pipeline.fit(X, y)

# 7. 获取特征名称
# 获取数值特征名
feature_names = numeric_features.copy()

# 获取分类特征名（独热编码后）
onehot_columns = pipeline.named_steps['preprocessor'].transformers_[1][1]\
                   .named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(onehot_columns)

# 8. 获取特征重要性
importances = pipeline.named_steps['classifier'].feature_importances_
std = np.std([tree.feature_importances_ for tree in pipeline.named_steps['classifier'].estimators_], axis=0)

# 9. 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
    'Std': std
}).sort_values('Importance', ascending=False)

# 10. 绘制特征重要性图（前22个特征）
plt.figure(figsize=(12, 10))
top_features = feature_importance_df.head(22)
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('绘制的图片/rf_feature_importance83.png', dpi=600)
plt.show()

# 11. 绘制特征重要性图（所有特征）
plt.figure(figsize=(10, 18))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('All Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('绘制的图片/rf_all_feature_importance.png', dpi=300)
plt.show()

# 12. 输出特征重要性
print("="*80)
print("Top 22 Features by Importance:")
print(top_features[['Feature', 'Importance']].to_string(index=False))
print("="*80)

# 13. 保存结果到Excel
with pd.ExcelWriter('生成的表格/rf_feature_importance_results83.xlsx') as writer:
    feature_importance_df.to_excel(writer, sheet_name='All Features', index=False)
    top_features.to_excel(writer, sheet_name='Top 22 Features', index=False)

# 14. 保存模型
joblib.dump(pipeline, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")

# 15. 输出最重要的特征变量（>0.01或>0.05)
selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature']
print("\nSelected Features (Importance > 0.01):")
print(selected_features.to_string(index=False))
#得到RF大于0.01的特征

#Lasso回归特征变量筛选（见R语言）

#SVM-RFE、Lasso和RF特征变量求交集并绘制韦恩图
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# 定义三个特征选择方法得到的特征集合
lasso_features = {'Occupation', 'Gender', 'Mechanical ventilation', 'Urinary catheterization', 'ICU admission', 'Mechanical ventilation (days)',
                  'Central venous catheter (CVC)', 'Disease type', 'Bacterial infection', 'Restricted antimicrobial use', 'Hospital stay (days)',
                  'Non-restricted antimicrobial use', 'Special-class antimicrobial use', 'Antimicrobial use', 'Duration of isolation',
                  'Combination antimicrobial therapy', 'Isolation order', 'Fever status', 'Elderly', 'WBC', 'CRP', 'PCT'
                  }

svm_features = {'Urinary catheterization', 'Mechanical ventilation', 'Restricted antimicrobial use', 'Central venous catheter (CVC)', 'Combination antimicrobial therapy', 'Antimicrobial use',
                'Special-class antimicrobial use', 'Bacterial infection',
                'Fever status', 'Disease type'
                }
rf_features = {'Hospital stay (days)', 'Antimicrobial use', 'WBC', 'Restricted antimicrobial use', 'Elderly', 'Mechanical ventilation (days)',
               'Combination antimicrobial therapy', 'PCT', 'Bacterial infection', 'Urinary catheterization (days)', 'Fever status',
               'IL6', 'Special-class antimicrobial use', 'Central venous catheter (CVC) (days)', 'Non-restricted antimicrobial use', 
               'CRP', 'Urinary catheterization', 'Central venous catheter (CVC)', 'Surgery', 'Gender', 'Mechanical ventilation',
               'Disease type'
               }

# 计算交集
common_features = lasso_features & svm_features & rf_features
print("交集特征:", common_features)

plt.figure(figsize=(10, 8))
venn = venn3([lasso_features, svm_features, rf_features], 
             set_labels=('Lasso Features', 'SVM Features', 'Random Forest Features'),
             set_colors=('#FF9999', '#66B2FF', '#99FF99'), 
             alpha=0.7)

# 设置字体大小和样式
for text in venn.set_labels:
    if text: text.set_fontsize(14)
for text in venn.subset_labels:
    if text: text.set_fontsize(12)

# 添加标题
plt.title("Feature Selection Comparison", fontsize=16, pad=20)

# 显示图形
plt.tight_layout()
plt.savefig('绘制的图片/韦恩图862.png', dpi=600)
plt.show()

# 创建韦恩图
plt.figure(figsize=(10, 8))
venn = venn3(
    subsets=(lasso_features, svm_features, rf_features),
    set_labels=('LASSO', 'SVM', 'RF'),
    set_colors=('#1f77b4', '#ff7f0e', '#2ca02c'),
    alpha=0.7
)

# 设置文本属性和标题
for text in venn.set_labels:
    text.set_fontsize(14)
for text in venn.subset_labels:
    if text: text.set_fontsize(12)
    
plt.title("Feature Selection Methods Comparison", fontsize=16, pad=20)

# 添加图例显示交集特征
if common_features:
    plt.text(-1.8, -1.2, f"Common Features: {', '.join(common_features)}", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('绘制的图片/韦恩图86.png', dpi=600)
plt.show()


#方法二韦恩图
# 计算交集
common_features = lasso_features & svm_features & rf_features

# 输出交集结果
print("="*50)
print("特征选择结果：")
print(f"LASSO选择的特征: {', '.join(lasso_features)}")
print(f"SVM选择的特征: {', '.join(svm_features)}")
print(f"RF选择的特征: {', '.join(rf_features)}")
print("-"*50)
print(f"交集特征({len(common_features)}个): {', '.join(common_features)}")
print("="*50)

# 创建韦恩图
plt.figure(figsize=(10, 8))
venn = venn3(
    subsets=(lasso_features, svm_features, rf_features),
    set_labels=('LASSO', 'SVM', 'RF'),
    set_colors=('#1f77b4', '#ff7f0e', '#2ca02c'),  # 蓝色、橙色、绿色
    alpha=0.7
)

# 设置文本属性
plt.title("Feature Selection Methods Comparison", fontsize=16, pad=20, fontweight='bold')
plt.suptitle("LASSO vs SVM vs RF", 
             fontsize=12, y=0.92, color='gray')

# 添加图例显示交集特征
if common_features:
    plt.text(0.5, -0.15, f"共同特征: {', '.join(common_features)}", 
             fontsize=12, ha='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 调整布局并显示
plt.tight_layout()
plt.savefig('绘制的图片/feature_selection_venn832.png', dpi=600, bbox_inches='tight')
plt.show()

#最终得到9个特征变量：Antimicrobial use, Urinary catheterization, Fever status, Combination antimicrobial therapy, Special-class antimicrobial use, Restricted antimicrobial use, 
# Bacterial infection, Central venous catheter (CVC), Disease type, Mechanical ventilation
###################################################################################################################
########### 基于训练集构建预测模型并基于验证集调优模型超参数（测试集数据应当只参与外部验证，其他地方不能用到） #############
###################################################################################################################
############################# 1. Logistic模型 ##############################
########### 线性模型不涉及超参数调优，我们用训练集训练模型后内部验证 ############
#lasso回归筛选出有意义变量
train_data = pd.read_csv("数据/train_data_notscaled.csv") # 训练集数据
train_data['Disease type'] = train_data['Disease type'].astype('category')
print(train_data.info())  # 查看数据结构
significant_vars_lasso=["WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy", "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation", "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection", "Restricted antimicrobial use", "Special-class antimicrobial use"]

# 构建预测模型 
X_train = train_data[significant_vars_lasso] # 提取训练集中显著的自变量
X_train_const = sm.add_constant(X_train) # 训练集添加常数项（截距）
y_train = train_data['Fungal infection'] # 结局变量
# 训练模型 
logist_model = sm.Logit(y_train, X_train_const).fit(disp=0)  # 关闭迭代信息
logist_model.summary() # 查看模型信息
# 计算训练集AUC，浅看模型效果
y_train_pred_prob_logist = logist_model.predict(X_train_const) # 在训练集预测所有个体的结局发生概率
auc_logist = roc_auc_score(y_train, y_train_pred_prob_logist) # 计算AUC
auc_logist

# 保存训练好的Logistic模型
with open("训练好的模型/logistic_model_Lo.pkl", 'wb') as f:
    pickle.dump(logist_model, f)


############################### 2. 决策树模型(可不用标准化数据) ###############################
# 导入标准化的训数据集
train_data_scaled = pd.read_csv("数据/train_data_notscaled.csv")
train_data_scaled['Disease type'] = train_data_scaled['Disease type'].astype('category')
print(train_data_scaled.info())  # 查看数据结构
# 提取自变量和结局变量
X = train_data_scaled[["WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy", "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation", 
                       "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection", "Restricted antimicrobial use", "Special-class antimicrobial use"]]
y = train_data_scaled['Fungal infection'] 
# 2.1. 将原始训练数据集再次划分训练集和验证集（可以理解为内部测试集）（7:3）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# 2.2. 使用默认参数训练CART决策树
tree_default = DecisionTreeClassifier(random_state=123) # 创建决策树分类模型，使用默认参数
tree_default.fit(X_train, y_train) # 在训练集上拟合决策树模型
# 查看模型的默认参数
print("模型默认参数:", pd.DataFrame.from_dict(tree_default.get_params(),orient='index'))
# 2.3. 计算默认参数模型的验证集AUC
y_val_pred_prob_treed = tree_default.predict_proba(X_val)[:, 1] # 验证集预测结局概率
auc_treed = roc_auc_score(y_val, y_val_pred_prob_treed) # 计算验证集AUC
print("默认参数模型的验证集 AUC:", auc_treed)
## 进行模型超参数的网格搜索调优模型超参数（基于验证集 AUC 最高） ##
# 2.4. 定义超参数搜索范围
param_grid = {
    'max_depth': [3, 5, None], # 树的最大深度
    'min_samples_split': [ 60, 80, 100], # 节点至少需要多少个样本，才会继续分裂
    'max_features': ['sqrt', None], # 在每次分裂时，决策树可以考虑的最大特征数
    'ccp_alpha': [0.0, 0.01, 0.1] # 剪枝时的复杂度惩罚系数
}
# 2.5. 使用网格搜索同时优化这些超参数
best_auc_tree = 0  # 记录最高 AUC
tree_model_best = None # 用于记录最佳决策树模型（最佳超参数）
best_max_depth = None  # 用于记录最佳 max_depth (最佳树深度)
best_min_samples_split = None  # 用于记录最佳 min_samples_split
best_max_features = None  # 用于记录最佳 max_features
best_ccp_alpha = None  # 用于记录最佳 ccp_alpha
for max_depth in param_grid['max_depth']:
    for min_samples_split in param_grid['min_samples_split']:
         for max_features in param_grid['max_features']:
              for ccp_alpha in param_grid['ccp_alpha']:
                # 设定决策树模型超参数
                tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                                    max_features=max_features, ccp_alpha=ccp_alpha,
                                                    random_state=123)
                # 训练模型
                tree_model.fit(X_train, y_train)
                # 在验证集上计算 AUC
                y_val_pred_prob = tree_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_prob)
                # 记录最优参数
                if auc > best_auc_tree:
                    tree_model_best = tree_model
                    best_auc_tree = auc
                    best_max_depth = max_depth
                    best_min_samples_split = min_samples_split
                    best_max_features = max_features
                    best_ccp_alpha = ccp_alpha
# 2.6. 输出最优超参数
print("最佳参数组合: max_depth =", best_max_depth, ", min_samples_split, =", best_min_samples_split, 
        ", max_features, =", best_max_features, ", ccp_alpha, =", best_ccp_alpha)
# 2.7. 验证集AUC对比：参数调优决策树 vs 默认参数决策树
print("默认参数决策树模型的验证集 AUC:", auc_treed)
print("参数调优决策树模型的验证集 AUC:", best_auc_tree)
print("调优模型参数:", pd.DataFrame.from_dict(tree_model_best.get_params(),orient='index'))
# 2.8. 绘制决策树
plt.figure(figsize=(10, 5))
plot_tree(tree_model_best, feature_names=X.columns, class_names=['NO-FI', 'FI'], filled=True)
plt.savefig("绘制的图片/tree_structure525.jpg", dpi=500)
plt.show()
# 保存训练好的模型
with open("训练好的模型/tree_model.pkl", 'wb') as f:
    pickle.dump(tree_model_best, f)

################### 3. 随机森林(RF)模型 ##########################
# 3.1. 使用默认参数训练随机森林模型
rf_model_default = RandomForestClassifier(random_state=123, oob_score=True) # 默认使用100棵数，最大特征数为变量数的开方（这里为4）
rf_model_default.fit(X_train, y_train)
# 查看模型默认参数
print("模型默认参数:", pd.DataFrame.from_dict(rf_model_default.get_params(),orient='index'))
# 3.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("默认参数模型的验证集 AUC:", auc_rfd)
# 进行超参数网格搜索调优（基于验证集 AUC 最高） #
# 3.3. 定义超参数搜索范围
param_grid = {
    'n_estimators': np.arange(50, 250, 50),  # 树的数量：50 到 300，每次增加 50
    'max_features': list(range(2, round(np.sqrt(X.shape[1])) + 1))  # 每棵树的最大特征使用数：2 到 自变量数
}
# 3.4. 使用网格搜索同时优化这些超参数
best_auc_rf = 0
rf_model_best = None
best_params_rf = {}
# 遍历所有超参数组合
for n_estimators in param_grid['n_estimators']:
    for max_features in param_grid['max_features']:
        # 定义随机森林模型
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_features=max_features,
            random_state=123,
            n_jobs=-1
        )
        # 训练模型
        rf_model.fit(X_train, y_train)
        # 计算验证集AUC
        y_val_pred_prob = rf_model.predict_proba(X_val)[:, 1]   #1表示第二列，关注结局发生的概率，因为结局是否发生定义为：否=0，是=1
        auc = roc_auc_score(y_val, y_val_pred_prob)
        # 记录最佳参数组合
        if auc > best_auc_rf:
            best_auc_rf = auc
            rf_model_best = rf_model
            best_params_rf = {
                'n_estimators': n_estimators,
                'max_features': max_features
            }
# 3.4. 输出最优超参数
best_ntree = best_params_rf['n_estimators']
best_mtry = best_params_rf['max_features']
print("最佳参数组合: n_estimators =", best_ntree, ", max_features =", best_mtry)
# 3.5. 输出最优超参数组合
print("最佳RF参数组合:", best_params_rf)
print("默认参数RF模型的验证集 AUC:", auc_rfd)
print("参数调优RF模型的验证集 AUC:", best_auc_rf)
print("调优RF模型参数:", pd.DataFrame.from_dict(rf_model_best.get_params(), orient='index'))
# 3.6. 保存训练好的模型
with open("训练好的模型/rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)

###################### 4. Xgboost模型 ##########################
# 4.1. 使用默认参数训练XGBoost
xgb_default = XGBClassifier(random_state=123, use_label_encoder=False, eval_metric='logloss', enable_categorical='True')
xgb_default.fit(X_train, y_train)
# 4.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_xgbd = xgb_default.predict_proba(X_val)[:, 1]
auc_xgbd = roc_auc_score(y_val, y_val_pred_prob_xgbd)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
# 4.3. 定义超参数搜索范围
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'max_depth': [3, 5, 10],  # 最大深度
    'n_estimators': [50, 100, 200],  # 弱分类器数量
    'subsample': [0.6, 0.8, 1.0]  # 采样比例
}
# 4.4. 进行网格搜索调优
best_auc_xgb = 0
xgb_model_best = None
best_params_xgb = {}
for learning_rate in param_grid['learning_rate']:
    for max_depth in param_grid['max_depth']:
        for n_estimators in param_grid['n_estimators']:
            for subsample in param_grid['subsample']:
                # 定义XGBoost模型
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
                # 训练模型
                xgb_model.fit(X_train, y_train)
                # 计算验证集AUC
                y_val_pred_prob = xgb_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_prob)
                # 记录最佳参数组合
                if auc > best_auc_xgb:
                    best_auc_xgb = auc
                    xgb_model_best = xgb_model
                    best_params_xgb = {
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'subsample': subsample
                    }
# 4.5. 输出最优超参数组合
print("最佳XGBoost参数组合:", best_params_xgb)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
print("参数调优XGBoost模型的验证集 AUC:", best_auc_xgb)
print("调优XGBoost模型参数:", pd.DataFrame.from_dict(xgb_model_best.get_params(), orient='index'))
# 4.6. 保存训练好的模型
with open("训练好的模型/xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)



import joblib
# 保存模型
joblib.dump(xgb_model_best, 'xgb_model.pkl')



###################### 5. LightGBM模型 ##########################
# 5.1. 使用默认参数训练LightGBM
lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)
# 5.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)
# 5.3. 定义超参数搜索范围
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'num_leaves': [31, 50, 100],  # 叶子节点数
    'n_estimators': [50, 100, 200],  # 弱分类器数量
    'subsample': [0.6, 0.8, 1.0],  # 采样比例
    'colsample_bytree': [0.6, 0.8, 1.0]  # 特征子集采样比例
}
# 5.4. 进行网格搜索调优
best_auc_lgb = 0
lgb_model_best = None
best_params_lgb = {}
for learning_rate in param_grid['learning_rate']:
    for num_leaves in param_grid['num_leaves']:
        for n_estimators in param_grid['n_estimators']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    # 定义LightGBM模型
                    lgb_model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        n_estimators=n_estimators,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=123
                    )
                    
                    # 训练模型
                    lgb_model.fit(X_train, y_train)
                    
                    # 计算验证集AUC
                    y_val_pred_prob = lgb_model.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, y_val_pred_prob)
                    
                    # 记录最佳参数组合
                    if auc > best_auc_lgb:
                        best_auc_lgb = auc
                        lgb_model_best = lgb_model
                        best_params_lgb = {
                            'learning_rate': learning_rate,
                            'num_leaves': num_leaves,
                            'n_estimators': n_estimators,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree
                        }
# 5.5. 输出最优超参数组合
print("最佳LightGBM参数组合:", best_params_lgb)
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)
print("参数调优LightGBM模型的验证集 AUC:", best_auc_lgb)
print("调优LightGBM模型参数:", pd.DataFrame.from_dict(lgb_model_best.get_params(), orient='index'))
# 5.6. 保存训练好的模型
with open("训练好的模型/lgb_model.pkl", 'wb') as f:
    pickle.dump(lgb_model_best, f)


################################################################################################
################################## 验证数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
# 得到训练集数据（对于lasso模型）
train_data = pd.read_csv("数据/train_data_notscaled.csv")
print(train_data.info()) 
train_data['Disease type'] = train_data['Disease type'].astype('category')
significant_vars_lasso=["WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy", "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation", 
                       "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection", "Restricted antimicrobial use", "Special-class antimicrobial use"]
X_train_logist = train_data[significant_vars_lasso]
X_train_logist_const = sm.add_constant(X_train_logist)
y_train_logist = train_data['Fungal infection']
# 得到验证数据集（对于机器学习模型）
train_data_scaled = pd.read_csv("数据/train_data_notscaled.csv")
train_data_scaled['Disease type'] = train_data_scaled['Disease type'].astype('category')
X = train_data_scaled[["WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy", "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation", 
                       "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection", "Restricted antimicrobial use", "Special-class antimicrobial use"]]
y = train_data_scaled['Fungal infection'] 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                  random_state=123, stratify=y) # 数据拆分时种子数必须和之前训练时的相同
###################### 1. 加载训练好的模型 ##########################
# 1.1. Logistic模型
with open("训练好的模型/logistic_model_Lo.pkl", 'rb') as f:
    logist_model = pickle.load(f)
# 1.2. 决策树模型
with open("训练好的模型/tree_model.pkl", 'rb') as f:
    tree_model = pickle.load(f)
# 1.3. 随机森林模型
with open("训练好的模型/rf_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)
# 1.4. XGBoost模型
with open("训练好的模型/xgb_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)
# 1.5. LightGBM模型
with open("训练好的模型/lgb_model.pkl", 'rb') as f:
    lgb_model = pickle.load(f)

## run end ##

###################### 2. 得到验证数据集预测结果，包括预测概率和预测分类 ##########################
# 2.1. Logistic模型（该模型为训练集评价，因为不涉及调参，也就没有验证集）
y_train_pred_prob_logist = logist_model.predict(X_train_logist_const) # 预测概率
y_train_pred_logist = (y_train_pred_prob_logist >= 0.5).astype(int) # 预测分类值（阈值0.5）
# 1.2. 决策树模型
y_val_pred_prob_tree = tree_model.predict_proba(X_val)[:, 1]
y_val_pred_tree = (y_val_pred_prob_tree >= 0.5).astype(int)
# 2.3. 随机森林模型
y_val_pred_prob_rf = rf_model.predict_proba(X_val)[:, 1]
y_val_pred_rf = (y_val_pred_prob_rf >= 0.5).astype(int)
# 2.4. XGBoost模型
y_val_pred_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred_xgb = (y_val_pred_prob_xgb >= 0.5).astype(int)
# 2.5. LightGBM模型
y_val_pred_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
y_val_pred_lgb = (y_val_pred_prob_lgb >= 0.5).astype(int)


###################### 3. 计算混淆矩阵并可视化 ##########################
## 编写混淆矩阵可视化函数，方便调用 ##
def CM_plot(cm):
    plt.figure(figsize=(5, 4)) # 可视化
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No FI', 'FI'], yticklabels=['No FI', 'FI'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
# 3.1. Logistic模型
cm_logist = confusion_matrix(y_train_logist, y_train_pred_logist) # 混淆矩阵
cm_logist
CM_plot(cm_logist)
# 3.2. 决策树模型
cm_tree = confusion_matrix(y_val, y_val_pred_tree)
CM_plot(cm_tree)
# 3.3. 随机森林模型
cm_rf = confusion_matrix(y_val, y_val_pred_rf)
CM_plot(cm_rf)
# 3.4. XGBoost模型
cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
CM_plot(cm_xgb)
# 3.5. LightGBM模型
cm_lgb = confusion_matrix(y_val, y_val_pred_lgb)
CM_plot(cm_lgb)


###################### 4. 计算准确率、精确率、灵敏度、f1分数、特异度 ##########################
def calculate_acc_pre_sen_f1_spc(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + fn + tn + fp) # 计算准确率（Accuracy rate）
    precision = tp / (tp + fp) # 计算精确率（Precision）
    sensitivity = tp / (tp + fn) # 计算灵敏度（Sensitivity, 召回率）
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = tn / (tn + fp) # 计算特异度（Specificity）
    return accuracy, precision, sensitivity, f1score, specificity
# 4.1. Logistic模型
accuracy_logist, precision_logist, sensitivity_logist, f1_logist, specificity_logist = calculate_acc_pre_sen_f1_spc(cm_logist)
print(f"Logistic Model → Accuracy: {accuracy_logist:.3f}, Precision: {precision_logist:.3f}, Sensitivity: {sensitivity_logist:.3f}, F1 Score: {f1_logist:.3f}, Specificity: {specificity_logist:.3f}")
# 4.2. 决策树模型
accuracy_tree, precision_tree, sensitivity_tree, f1_tree, specificity_tree = calculate_acc_pre_sen_f1_spc(cm_tree)
print(f"Decision Tree Model → Accuracy: {accuracy_tree:.3f}, Precision: {precision_tree:.3f}, Sensitivity: {sensitivity_tree:.3f}, F1 Score: {f1_tree:.3f}, Specificity: {specificity_tree:.3f}")
# 4.3. 随机森林模型
accuracy_rf, precision_rf, sensitivity_rf, f1_rf, specificity_rf = calculate_acc_pre_sen_f1_spc(cm_rf)
print(f"Random Forest Model → Accuracy: {accuracy_rf:.3f}, Precision: {precision_rf:.3f}, Sensitivity: {sensitivity_rf:.3f}, F1 Score: {f1_rf:.3f}, Specificity: {specificity_rf:.3f}")
# 4.4. XGBoost模型
accuracy_xgb, precision_xgb, sensitivity_xgb, f1_xgb, specificity_xgb = calculate_acc_pre_sen_f1_spc(cm_xgb)
print(f"XGBoost Model → Accuracy: {accuracy_xgb:.3f}, Precision: {precision_xgb:.3f}, Sensitivity: {sensitivity_xgb:.3f}, F1 Score: {f1_xgb:.3f}, Specificity: {specificity_xgb:.3f}")
# 4.5. LightGBM模型
accuracy_lgb, precision_lgb, sensitivity_lgb, f1_lgb, specificity_lgb = calculate_acc_pre_sen_f1_spc(cm_lgb)
print(f"LightGBM Model → Accuracy: {accuracy_lgb:.3f}, Precision: {precision_lgb:.3f}, Sensitivity: {sensitivity_lgb:.3f}, F1 Score: {f1_lgb:.3f}, Specificity: {specificity_lgb:.3f}")



###################### 5. 计算AUC及其95%置信区间 ##########################
## 编写AUC及其95%置信区间计算的函数 ##
def calculate_auc(y_label, y_pred_prob):
    auc_value = roc_auc_score(y_label, y_pred_prob)
    se_auc = np.sqrt((auc_value * (1 - auc_value)) / len(y_label))
    z = norm.ppf(0.975)  # 95% CI 的z值
    auc_ci_lower = auc_value - z * se_auc
    auc_ci_upper = auc_value + z * se_auc
    return auc_value, auc_ci_lower, auc_ci_upper
# 5.1. Logistic模型
auc_value_logist, auc_ci_lower_logist, auc_ci_upper_logist = calculate_auc(y_train_logist, y_train_pred_prob_logist)
print(f"Logistic Model AUC: {auc_value_logist:.3f} (95% CI: {auc_ci_lower_logist:.3f} - {auc_ci_upper_logist:.3f})")
# 5.2. 决策树模型
auc_value_tree, auc_ci_lower_tree, auc_ci_upper_tree = calculate_auc(y_val, y_val_pred_prob_tree)
print(f"Decision Tree Model AUC: {auc_value_tree:.3f} (95% CI: {auc_ci_lower_tree:.3f} - {auc_ci_upper_tree:.3f})")
# 5.3. 随机森林模型
auc_value_rf, auc_ci_lower_rf, auc_ci_upper_rf = calculate_auc(y_val, y_val_pred_prob_rf)
print(f"Random Forest Model AUC: {auc_value_rf:.3f} (95% CI: {auc_ci_lower_rf:.3f} - {auc_ci_upper_rf:.3f})")
# 5.4. XGBoost模型
auc_value_xgb, auc_ci_lower_xgb, auc_ci_upper_xgb = calculate_auc(y_val, y_val_pred_prob_xgb)
print(f"XGBoost Model AUC: {auc_value_xgb:.3f} (95% CI: {auc_ci_lower_xgb:.3f} - {auc_ci_upper_xgb:.3f})")
# 5.5. LightGBM模型
auc_value_lgb, auc_ci_lower_lgb, auc_ci_upper_lgb = calculate_auc(y_val, y_val_pred_prob_lgb)
print(f"LightGBM Model AUC: {auc_value_lgb:.3f} (95% CI: {auc_ci_lower_lgb:.3f} - {auc_ci_upper_lgb:.3f})")


###################### 6. 绘制ROC曲线 ##########################
## 编写绘制ROC曲线的函数 ##
def ROC_plot(y_label, y_pred_prob,auc_value):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.show()
# 6.1. Logistic模型
ROC_plot(y_train_logist, y_train_pred_prob_logist, auc_value_logist)
# 6.2. 决策树模型
ROC_plot(y_val, y_val_pred_prob_tree, auc_value_tree)
# 6.3. 随机森林模型
ROC_plot(y_val, y_val_pred_prob_rf, auc_value_rf)
# 6.4. XGBoost模型
ROC_plot(y_val, y_val_pred_prob_xgb, auc_value_xgb)
# 6.5. LightGBM模型
ROC_plot(y_val, y_val_pred_prob_lgb, auc_value_lgb)


###################### 7. 绘制校准曲线 ##########################
## 编写绘制校准曲线的函数 ##
def CaliC_plot(y_label, y_pred_prob,n_bins=10):
    prob_true, prob_pred = calibration_curve(y_label, y_pred_prob, n_bins=n_bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()
# 7.1. Logistic模型
CaliC_plot(y_train_logist, y_train_pred_prob_logist)
CaliC_plot(y_train_logist, y_train_pred_prob_logist, n_bins=8)
# 7.2. 决策树模型
CaliC_plot(y_val, y_val_pred_prob_tree) # 出现原因是数据异质性不高
# 7.3. 随机森林模型
CaliC_plot(y_val, y_val_pred_prob_rf) # 预测概率高的人少+数据异质性不高，人数多就好了
# 7.4. XGBoost模型
CaliC_plot(y_val, y_val_pred_prob_xgb)
# 7.5. LightGBM模型
CaliC_plot(y_val, y_val_pred_prob_lgb)


###################### 8. 绘制决策分析曲线 (DCA) ##########################
## 编写计算净收益的函数，方便调用 ##
def calculate_net_benefi(y_label, y_pred_prob,
                                thresholds = np.linspace(0.01, 1, 100)):
    net_benefit_model = [] # 用于保存在不同阳性阈值下基于模型的净收益
    net_benefit_alltrt = [] # 用于保存在不同阳性阈值下假定所有人都接受治疗时的净收益
    net_benefits_notrt = [0] * len(thresholds) # 假定所有人都不接受治疗时的净收益，始终为0
    total_obs = len(y_label)
    for thresh in thresholds:
        # 对于基于模型的净收益
        y_pred_label = y_pred_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        net_benefit = (tp / total_obs) - (fp / total_obs) * (thresh / (1 - thresh))
        net_benefit_model.append(net_benefit)
        # 对于假定所有人都接受治疗时的净收益
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total_right = tp + tn
        net_benefit = (tp / total_right) - (tn / total_right) * (thresh / (1 - thresh))
        net_benefit_alltrt.append(net_benefit)
    return net_benefit_model, net_benefit_alltrt, net_benefits_notrt
## 编写绘制DCA曲线的函数，方便调用 ##
def DCA_plot(net_benefit_model, net_benefit_alltrt, net_benefits_notrt,
             thresholds = np.linspace(0.01,0.99,100)):
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, net_benefit_model, label="Model Net Benefit", color='blue', linewidth=2)
    plt.plot(thresholds, net_benefit_alltrt, label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefits_notrt, linestyle='--', color='green', label="Treat None", linewidth=2)
    plt.xlabel("Threshold Probability")
    plt.ylim(-0.10,np.nanmax(np.array(net_benefit_model))+0.05)
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()
# 8.1. Logistic模型
net_benefit_logist, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_train_logist, y_train_pred_prob_logist)
DCA_plot(net_benefit_logist, net_benefit_alltrt, net_benefits_notrt)
# 8.2. 决策树模型
net_benefit_tree, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_tree)
DCA_plot(net_benefit_tree, net_benefit_alltrt, net_benefits_notrt)
# 8.3. 随机森林模型
net_benefit_rf, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_rf)
DCA_plot(net_benefit_rf, net_benefit_alltrt, net_benefits_notrt)
# 8.4. XGBoost模型
net_benefit_xgb, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_xgb)
DCA_plot(net_benefit_xgb, net_benefit_alltrt, net_benefits_notrt)
# 8.5. LightGBM模型
net_benefit_lgb, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_lgb)
DCA_plot(net_benefit_lgb, net_benefit_alltrt, net_benefits_notrt)


###################### 9. 所有模型的训练集/验证集预测效果汇总，方便对比 ##########################
# 9.1. 汇总模型评估指标并保存 
model_results_validation = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
    "AUC": [auc_value_logist, auc_value_tree, auc_value_rf, auc_value_xgb, auc_value_lgb],
    "AUC 95% CI Lower": [auc_ci_lower_logist, auc_ci_lower_tree, auc_ci_lower_rf, auc_ci_lower_xgb, auc_ci_lower_lgb],
    "AUC 95% CI Upper": [auc_ci_upper_logist, auc_ci_upper_tree, auc_ci_upper_rf, auc_ci_upper_xgb, auc_ci_upper_lgb],
    "Accuracy": [accuracy_logist, accuracy_tree, accuracy_rf, accuracy_xgb, accuracy_lgb],
    "Precision": [precision_logist, precision_tree, precision_rf, precision_xgb, precision_lgb],
    "Sensitivity": [sensitivity_logist, sensitivity_tree, sensitivity_rf, sensitivity_xgb, sensitivity_lgb],
    "Specificity": [specificity_logist, specificity_tree, specificity_rf, specificity_xgb, specificity_lgb],
    "F1 Score": [f1_logist, f1_tree, f1_rf, f1_xgb, f1_lgb]
}) # 创建DataFrame存储结果
model_results_validation
model_results_validation.to_csv("预测效果评价文件/model_performance_validation832.csv", index=False) # 保存为CSV文件
# 9.2. 在一张图上绘制所有模型的ROC曲线 #
plt.figure(figsize=(8, 6))
models = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist, auc_value_logist),
    "Decision Tree": (y_val, y_val_pred_prob_tree, auc_value_tree),
    "Random Forest": (y_val, y_val_pred_prob_rf, auc_value_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb, auc_value_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb, auc_value_lgb),
}
for model_name, (y_true, y_pred_prob, auc_value) in models.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})") # 逐个绘制每个模型的ROC曲线
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("预测效果评价文件/ROC_curves_allmodel_validation832.png", dpi=600)
plt.show()
# 9.3. 在一张图上绘制所有模型的校准曲线 
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name) # 逐个绘制每个模型的校准曲线
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves for All Models")
plt.legend(loc='upper left')
plt.grid()
plt.savefig("预测效果评价文件/Calibration_curves_allmodel_validation832.png", dpi=600)
plt.show()
# 9.4. 在一张图上绘制所有模型的DCA曲线 
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_true, y_pred_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), net_benefit, label=model_name) # 逐个绘制DCA曲线
plt.plot(np.linspace(0.01, 0.99, 100), net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), net_benefits_notrt, linestyle="--", color="green", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylim(-0.10,np.nanmax(np.array(net_benefit))+0.05)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("预测效果评价文件/DCA_curves_allmodel_validation832.png", dpi=600)
plt.show()


################################################################################################
################################## 测试数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
# 得到测试集数据（Lasso模型）
test_data = pd.read_csv("数据/test_data_notscaled.csv")
test_data['Disease type'] = test_data['Disease type'].astype('category')
significant_vars_lasso=["WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy", "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation", 
                       "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection", "Restricted antimicrobial use", "Special-class antimicrobial use"]
X_test_logist = test_data[significant_vars_lasso]
X_test_logist_const = sm.add_constant(X_test_logist)
y_test_logist = test_data['Fungal infection']
# 得到测试数据集（机器学习模型）
test_data_scaled = pd.read_csv("数据/test_data_notscaled.csv")
test_data_scaled['Disease type'] = test_data_scaled['Disease type'].astype('category')
X_test = test_data_scaled[["WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy", "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation", 
                       "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection", "Restricted antimicrobial use", "Special-class antimicrobial use"]]
y_test = test_data_scaled['Fungal infection'] 
## run end ##

###################### 1. 计算测试数据集预测结果 ##########################
# 1.1. Logistic模型
y_test_pred_prob_logist = logist_model.predict(X_test_logist_const) # 预测概率
y_test_pred_logist = (y_test_pred_prob_logist >= 0.5).astype(int) # 预测分类值（阈值0.5）
# 1.2. 决策树模型
y_test_pred_prob_tree = tree_model.predict_proba(X_test)[:, 1]
y_test_pred_tree = (y_test_pred_prob_tree >= 0.5).astype(int)
# 1.3. 随机森林模型
y_test_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_test_pred_rf = (y_test_pred_prob_rf >= 0.5).astype(int)
# 1.4. XGBoost模型
y_test_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred_xgb = (y_test_pred_prob_xgb >= 0.5).astype(int)
# 1.5. LightGBM模型
y_test_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_test_pred_lgb = (y_test_pred_prob_lgb >= 0.5).astype(int)


###################### 2. 计算混淆矩阵并可视化 ##########################
# 2.1. Logistic模型
cm_logist_test = confusion_matrix(y_test_logist, y_test_pred_logist)
CM_plot(cm_logist_test)
# 2.2. 决策树模型
cm_tree_test = confusion_matrix(y_test, y_test_pred_tree)
CM_plot(cm_tree_test)
# 2.3. 随机森林模型
cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)
CM_plot(cm_rf_test)
# 2.4. XGBoost模型
cm_xgb_test = confusion_matrix(y_test, y_test_pred_xgb)
CM_plot(cm_xgb_test)
# 2.5. LightGBM模型
cm_lgb_test = confusion_matrix(y_test, y_test_pred_lgb)
CM_plot(cm_lgb_test)


###################### 3. 计算准确率、精确率、灵敏度、f1分数、特异度 ##########################
# 3.1. Logistic模型
accuracy_logist_test, precision_logist_test, sensitivity_logist_test, f1_logist_test, specificity_logist_test = calculate_acc_pre_sen_f1_spc(cm_logist_test)
# 3.2. 决策树模型
accuracy_tree_test, precision_tree_test, sensitivity_tree_test, f1_tree_test, specificity_tree_test = calculate_acc_pre_sen_f1_spc(cm_tree_test)
# 3.3. 随机森林模型
accuracy_rf_test, precision_rf_test, sensitivity_rf_test, f1_rf_test, specificity_rf_test = calculate_acc_pre_sen_f1_spc(cm_rf_test)
# 3.4. XGBoost模型
accuracy_xgb_test, precision_xgb_test, sensitivity_xgb_test, f1_xgb_test, specificity_xgb_test = calculate_acc_pre_sen_f1_spc(cm_xgb_test)
# 3.5. LightGBM模型
accuracy_lgb_test, precision_lgb_test, sensitivity_lgb_test, f1_lgb_test, specificity_lgb_test = calculate_acc_pre_sen_f1_spc(cm_lgb_test)


###################### 4. 计算AUC及其95%置信区间 ##########################
# 4.1. Logistic模型
auc_value_logist_test, auc_ci_lower_logist_test, auc_ci_upper_logist_test = calculate_auc(y_test_logist, y_test_pred_prob_logist)
# 4.2. 决策树模型
auc_value_tree_test, auc_ci_lower_tree_test, auc_ci_upper_tree_test = calculate_auc(y_test, y_test_pred_prob_tree)
# 4.3. 随机森林模型
auc_value_rf_test, auc_ci_lower_rf_test, auc_ci_upper_rf_test = calculate_auc(y_test, y_test_pred_prob_rf)
# 4.4. XGBoost模型
auc_value_xgb_test, auc_ci_lower_xgb_test, auc_ci_upper_xgb_test = calculate_auc(y_test, y_test_pred_prob_xgb)
# 4.5. LightGBM模型
auc_value_lgb_test, auc_ci_lower_lgb_test, auc_ci_upper_lgb_test = calculate_auc(y_test, y_test_pred_prob_lgb)


###################### 5. 所有模型的测试集预测效果汇总，方便对比 ##########################
model_results_test = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
    "AUC": [auc_value_logist_test, auc_value_tree_test, auc_value_rf_test, auc_value_xgb_test, auc_value_lgb_test],
    "AUC 95% CI Lower": [auc_ci_lower_logist_test, auc_ci_lower_tree_test, auc_ci_lower_rf_test, auc_ci_lower_xgb_test, auc_ci_lower_lgb_test],
    "AUC 95% CI Upper": [auc_ci_upper_logist_test, auc_ci_upper_tree_test, auc_ci_upper_rf_test, auc_ci_upper_xgb_test, auc_ci_upper_lgb_test],
    "Accuracy": [accuracy_logist_test, accuracy_tree_test, accuracy_rf_test, accuracy_xgb_test, accuracy_lgb_test],
    "Precision": [precision_logist_test, precision_tree_test, precision_rf_test, precision_xgb_test, precision_lgb_test],
    "Sensitivity": [sensitivity_logist_test, sensitivity_tree_test, sensitivity_rf_test, sensitivity_xgb_test, sensitivity_lgb_test],
    "Specificity": [specificity_logist_test, specificity_tree_test, specificity_rf_test, specificity_xgb_test, specificity_lgb_test],
    "F1 Score": [f1_logist_test, f1_tree_test, f1_rf_test, f1_xgb_test, f1_lgb_test]
})  # 创建DataFrame存储结果
model_results_test
model_results_test.to_csv("预测效果评价文件/model_performance_test83.csv", index=False)  # 保存为CSV文件

###################### 6. 绘制ROC曲线 ##########################
plt.figure(figsize=(8, 6))
models_test = {
    "Logistic": (y_test_logist, y_test_pred_prob_logist, auc_value_logist_test),
    "Decision Tree": (y_test, y_test_pred_prob_tree, auc_value_tree_test),
    "Random Forest": (y_test, y_test_pred_prob_rf, auc_value_rf_test),
    "XGBoost": (y_test, y_test_pred_prob_xgb, auc_value_xgb_test),
    "LightGBM": (y_test, y_test_pred_prob_lgb, auc_value_lgb_test),
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
plt.savefig("预测效果评价文件/ROC_curves_allmodel_test83.png", dpi=600)
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
plt.savefig("预测效果评价文件/Calibration_curves_allmodel_test83.png", dpi=600)
plt.show()

###################### 8. 绘制DCA曲线 ##########################
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models_test.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_true, y_pred_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), net_benefit, label=model_name)
plt.plot(np.linspace(0.01, 0.99, 100), net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), net_benefits_notrt, linestyle="--", color="green", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylim(-0.10,np.nanmax(np.array(net_benefit))+0.05)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models (Test Set)")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("预测效果评价文件/DCA_curves_allmodel_test83.png", dpi=600)
plt.show()


################################################################################################
################################## 模型的SHAP解释 ###############################################
################################################################################################
save_path = "SHAP结果文件/" # 图片保存路径
#SHAP可解释性，XGBOOST（最优模型）
# 导入标准化的训数据集
train_data_scaled = pd.read_csv("数据/train_data_notscaled.csv")
train_data_scaled['Disease type'] = train_data_scaled['Disease type'].astype('category')
# 提取自变量和结局变量
X = train_data_scaled[["WBC", "CRP", "IL6", "PCT", "Elderly", "Combination antimicrobial therapy", "Urinary catheterization", "Disease type", "Fever status", "Mechanical ventilation", 
                       "Central venous catheter (CVC)", "Antimicrobial use", "Bacterial infection", "Restricted antimicrobial use", "Special-class antimicrobial use"]]
y = train_data_scaled['Fungal infection'] 
# 2.1. 将原始训练数据集再次划分训练集和验证集（可以理解为内部测试集）（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
with open("训练好的模型/xgb_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)
# 使用下采样后的数据进行训练
xgb_model.fit(X_train, y_train)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["font.family"] = ["sans-serif"]
#plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sample = X_test.iloc[[9]]
# 计算 SHAP 值
explainer = shap.Explainer(xgb_model)
shap_values = explainer.shap_values(sample)
# 将 shap_values 转换为 Explanation 对象
# 将 shap_values 转换为单个解释对象
shap_exp = explainer(X_test.iloc[[9]])
# 绘制瀑布图
shap.waterfall_plot(shap_exp[0], max_display=22,show=False)
plt.xlabel('SHAP value (for an individual)', fontsize=14)
plt.gca().xaxis.set_label_coords(0.5, -0.1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.subplots_adjust(
    left=0.3,    # 增加左侧空间（用于y轴标签）
    right=0.95,  # 保留右侧空间
    top=0.95,    # 保留顶部空间
    bottom=0.2   # 增加底部空间（用于x轴标签）
)
# 旋转y轴标签（避免重叠）
for label in plt.gca().get_yticklabels():
    label.set_rotation(0)  # 水平显示标签
    label.set_ha('right')   # 右对齐

# 保存图像（先保存再显示）
plt.savefig("SHAP结果文件/xgboost的SHAP瀑布图832.png", 
            dpi=600, 
            bbox_inches='tight',  # 自动调整边界
            pad_inches=0.5)

plt.tight_layout() # 调整画布布局，避免边界内容显示不全
plt.show()

# 生成力图
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[[1]], matplotlib=True)
plt.show()
plt.savefig("SHAP结果文件/xgboost的热力图832.png", dpi=600)

# 从测试集中采样50个样本（确保样本多样性）
X_sample = X_test.sample(n=50, random_state=42)
# 初始化SHAP解释器（使用TreeExplainer加速）
explainer = shap.TreeExplainer(xgb_model)

# 计算50个样本的SHAP值（提升计算效率）
shap_values = explainer.shap_values(X_sample)

# 绘制特征重要性条形图（全局解释）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=22, show=False)
plt.title("XGBoost特征重要性（SHAP值）", fontsize=14, pad=20)
plt.xlabel("mean(SHAP value)(average impact on model output magnitude)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("SHAP结果文件/xgboost_SHAP特征重要性条形图83.png", dpi=600, bbox_inches='tight')
plt.show()

# 绘制特征蜂群图（展示分布）
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=22, show=False)
plt.title("XGBoost特征影响分布（SHAP值）", fontsize=14, pad=20)
plt.xlabel("SHAP value", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("SHAP结果文件/xgboost_SHAP蜂群图83.png", dpi=600, bbox_inches='tight')
plt.show()



