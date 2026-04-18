
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pathlib
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import xgboost
from xgboost import XGBClassifier

# 设置页面标题
st.set_page_config(
    page_title="Fungal infection prediction system", 
    page_icon="🩺",
    layout="wide"
)
st.title("Fungal infection prediction system")

# 加载模型
model = joblib.load('xgb_model.pkl')

# 侧边栏 - 关于
st.sidebar.header("About")
st.sidebar.info("""
Input the relevant parameters of the patient, and the system will calculate the probability of fungal infection occurrence.
""")

# 风险等级解释
st.sidebar.subheader("Risk level description")
st.sidebar.markdown("""
- **Low risk**: < 20% 
- **Concentration risk**: 20% - 50% 
- **High Risk**: > 50% 
""")

# 主界面
st.header("Please enter the patient parameters")

# 创建输入表单
with st.form("prediction_form"):
    # 使用多列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Laboratory index")
        # 实验室指标输入
        wbc = st.number_input(
            "White blood cell count (10^9/L)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help="Normal range: 4.0-10.0 × 10^9/L"
        )
        crp = st.number_input(
            "CRP (mg/L)",
            min_value=0.0,
            max_value=500.0,
            value=5.0,
            step=0.1
        )
        il6 = st.number_input(
            "IL6 (pg/mL)",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )
        pct = st.number_input(
            "PCT (ng/mL)",
            min_value=0.0,
            max_value=100.0,
            value=0.1,
            step=0.01
        )
        
    with col2:
        st.subheader("Patient characteristics")
        # 是否老龄选择
        elderly = st.selectbox(
            "Elderly(≥65 years old)",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        # 疾病分类选择
        disease_type = st.selectbox(
            "Disease type",
            options=[
                ("Others", 0),
                ("Rspiratory disease", 1),
                ("Tumour", 2),
                ("Gynecological disease", 3),
                ("Orthopedics and Trauma", 4),
                ("CVD", 6),
                ("Nerve system disease", 7),
                ("Digestive system disease", 8),
                ("Metabolic diseases", 9),
                ("Urinary system disease", 10),
                ("Infectious disease", 11),
                ("Otorhinolaryngological diseases", 12),
                ("Ophthalmic Diseases", 13),
                ("Dermatological diseases", 14),
                ("Hematological diseases", 15),
                ("Rehabilitation", 16),
                ("ICU", 17),
                ("Gereology", 18),
                ("General medicine", 19),
                ("Traditional Chinese Medicine and General Medicine", 20)
            ],
            format_func=lambda x: x[0]
        )[1]
        
        # 发热状态
        fever_status = st.selectbox(
            "Fever status",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        # 细菌感染
        bacterial_infection = st.selectbox(
            "Bacterial infection",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
    
    with col3:
        st.subheader("Medical intervention")
        # 抗菌药物使用相关
        antimicrobial_use = st.selectbox(
            "Antimicrobial use",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        urinary_catheter = st.selectbox(
            "Urinary catheterization",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        combination_therapy = st.selectbox(
            "Combination antimicrobial therapy",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        special_class_antimicrobial = st.selectbox(
            "Special-class antimicrobial use",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        restricted_antimicrobial = st.selectbox(
            "Restricted antimicrobial use",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        central_venous_catheter = st.selectbox(
            "Central venous catheter (CVC)",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        mechanical_ventilation = st.selectbox(
            "Mechanical ventilation",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0],
            help="Indicates whether the patient requires mechanical ventilation support"
        )[1]
    
    # 提交按钮
    submitted = st.form_submit_button("Predict the risk of fungal infections")
    
    if submitted:
        # 创建输入数据框 - 按照模型训练时的特征顺序
        input_data = pd.DataFrame([[
            wbc, crp, il6, pct, elderly, combination_therapy, urinary_catheter,
            disease_type, fever_status, mechanical_ventilation, central_venous_catheter, antimicrobial_use,
            bacterial_infection, restricted_antimicrobial, special_class_antimicrobial,
        ]], columns=[
            'WBC', 'CRP', 'IL6', 'PCT', 'Elderly', 'Combination antimicrobial therapy', 'Urinary catheterization',
            'Disease type', 'Fever status', 'Mechanical ventilation', 'Central venous catheter (CVC)', 'Antimicrobial use',
            'Bacterial infection',
            'Restricted antimicrobial use', 'Special-class antimicrobial use'
        ])
        
        # 预测概率
        try:
            proba = model.predict_proba(input_data)[0][1]  # 获取真菌感染的概率
            
            # 显示结果
            st.subheader("Predict the outcome")
            
            # 使用进度条和指标显示概率
            col_res1, col_res2 = st.columns([1, 3])
            with col_res1:
                st.metric(label="Probability of fungal infection", value=f"{proba:.1%}")
            with col_res2:
                st.progress(float(proba))
            
            # 风险等级评估
            if proba > 0.5:
                st.error("High risk")
            elif proba > 0.2:
                st.warning("Concentration risk")
            else:
                st.success("Low risk")
            
            # 详细解释
            st.markdown("### Clinical practice recommendations")
            if proba > 0.5:
                st.markdown("""
                - **Do it now**:
                  - Conduct fungal culture, G test and GM test
                  - Consider initiating empirical antifungal therapy
                  - Assess the immune status and underlying diseases
                
                - **Supervise**:
                  - Monitor the changes in infection indicators every day
                  - Evaluate the necessity of catheter use
                  - Review the use plan of antibacterial drugs
                """)
            elif proba > 0.2:
                st.markdown("""
                - **Further inspection**:
                  - Conduct laboratory tests related to fungi
                  - Assess the immune status and underlying diseases
                  - Regularly recheck infection indicators
                
                - **Preventive measures**:
                  - Evaluate the rationality of the use of antibacterial drugs
                  - Evaluate the necessity of catheter use
                  - Strengthen infection surveillance
                """)
            else:
                st.markdown("""
                - **Regular management**:
                  - Continue the current treatment plan
                  - Monitor changes in infection indicators
                  - If symptoms worsen, have a timely follow-up examination
                
                - **Prevention suggestions**:
                  - Use antibacterial drugs rationally
                  - Avoid unnecessary use of catheters
                  - Strengthen basic care
                """)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# 添加使用说明
st.markdown("---")
st.subheader("instructions")
st.markdown("""
1. Enter all the parameters of the patient in the form
2. The system will calculate and display the probability and risk level of fungal infection

**Parameter specification**:
- **Laboratory index**: The current laboratory test values of the patient
- **Patient characteristics**: The basic situation and disease information of the patient
- **Medical intervention**: The medical measures received by the patient and the use of drugs
""")

# 添加页脚
st.markdown("---")
st.caption("© Fungal infection prediction model")