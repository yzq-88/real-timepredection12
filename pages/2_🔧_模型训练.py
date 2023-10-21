import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置



st.set_page_config(layout="wide")
import os
if '是否训练' not in st.session_state:
    st.session_state['是否训练'] = 0
if 'block' not in st.session_state:
    st.session_state['block'] = ""
# 数据导入和预处理
@st.cache_resource()
def load_data(file):
    data = pd.read_excel(file)
    # 数据预处理，比如填充空值，数据归一化等。
    return data

def evaluate_and_display(model, X_train, y_train, X_test, y_test, model_name, training_time):
    # Fit model
    model.fit(X_train, y_train)

    # Predict test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Update session state
    st.session_state['是否训练'] = 1

    result_table = pd.DataFrame({
        '耗时（s）': [training_time],
        '训练集大小': [X_train.shape[0]],
        '测试集大小': [X_test.shape[0]],
        '平均平方误差': [mse],
        '平均绝对误差': [mae],
        '决定系数': [r2]
    })
    st.write(
        f'<style>div.row-widget.stRadio span div[role="radiogroup"] > label:nth-child(2) {{text-align: center;}}</style>',
        unsafe_allow_html=True,
    )
    st.dataframe(result_table)


def gen_model_name(model):
    current_time = datetime.now()
    current_time_text = current_time.strftime("%Y年%m月%d日%H:%M:%S")
    algorithm_name = model.__class__.__name__
    return algorithm_name+"-"+current_time_text
def plot_actual_vs_predicted(X_train, y_train, X_test, y_test, model):
    # Predict values for training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    # Plot for training data
    ax1.scatter(y_train, y_train_pred, color='blue', alpha=0.3)
    ax1.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
    ax1.set_title('训练集：真实值 vs 预测值')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')

    # Plot for test data
    ax2.scatter(y_test, y_test_pred, color='blue', alpha=0.3)
    ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    ax2.set_title('测试集：真实值 vs 预测值')
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('预测值')

    # Show plot
    plt.tight_layout()
    st.pyplot(fig)
with st.expander("1️⃣区块设置",expanded=True):
    directory_path = "./model"
    col1, col2 = st.columns(2)
    with col1:
        checkbox1 = st.checkbox("已有区块")
        if checkbox1:
            def get_subdirectories(directory_path):
                return [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

            subdirs = get_subdirectories(directory_path)
            selected_dir = st.selectbox("请选择应用区块:", subdirs)
            if selected_dir!="":
                MODELS_PATH = "./model/" + selected_dir
                st.session_state['block']=MODELS_PATH


    with col2:
        checkbox2 = st.checkbox("新增区块")
        if checkbox2:
            new_dir_name = st.text_input("新增区块名称（务必规范）")
            if new_dir_name:
                save_button = st.button("保存")
                if save_button:
                    new_dir_path = os.path.join(directory_path, new_dir_name)
                    if not os.path.exists(new_dir_path):
                        os.makedirs(new_dir_path)
                        st.success(f"区块 {new_dir_name} 已创建!")
                        st.experimental_rerun()  # 刷新页面
                    else:
                        st.warning(f"区块 {new_dir_name} 已存在!")

if st.session_state['block']!="":
    tab1,tab2=st.tabs(['模型训练','区块模型设置'])
    with tab1:
        col1,col2=st.columns(2)

        train_or_load = col1.radio("新建模型或者微调已存在模型：", ["新建模型", "微调模型"], horizontal=True)
        target=col2.selectbox("选择目标值", ["杨氏模量", "抗压强度", "泊松比", "内聚力", "内摩擦角", "孔隙压力"])
        uploaded_file = st.file_uploader("上传你的数据文件", type=['xlsx', 'csv'])


        if uploaded_file is not None:

            data = load_data(uploaded_file)
            feature = ['大勾载荷', '泵压', '机械钻速', '钻压', '转速', '排量', '密度','粘度']
            # 删除目标变量中的缺失行
            data = data.dropna(subset=[target])

            # 删除特征中缺失超过3个的筛选数据
            data = data.dropna(subset=feature, thresh=len(feature) - 3)

            X = data[feature]
            y = data[target]

            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if train_or_load == "新建模型":
                model_selection = st.selectbox("请选择模型：", ["Random Forest", "XGBoost", "LightGBM"])
                if model_selection == "Random Forest":
                    n_estimators = st.slider('n_estimators', min_value=10, max_value=500, value=100, step=10)
                    max_depth = st.slider('max_depth', min_value=10, max_value=100, value=30, step=10)
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                elif model_selection == "XGBoost":
                    max_depth = st.slider('max_depth', min_value=1, max_value=10, value=3, step=1)
                    learning_rate = st.slider('learning_rate', min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                    model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate)
                elif model_selection == "LightGBM":
                    num_leaves = st.slider('num_leaves', min_value=10, max_value=100, value=31, step=5)
                    learning_rate = st.slider('learning_rate', min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                    model = LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate)
                # 训练模型并展示结果
                if st.button("开始训练"):
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.session_state['是否训练'] = 1
                    evaluate_and_display(model, X_train, y_train, X_test, y_test, model_selection, training_time)

                    plot_actual_vs_predicted(X_train,y_train,X_test,y_test,model)
                if st.session_state['是否训练']==1:
                    co11,col2=st.columns(2)

                    model_name = co11.text_input("请输入模型名称：",value=target+'--'+gen_model_name(model),label_visibility='collapsed')
                    if col2.button("保存模型"):
                        st.session_state['是否训练'] = 2
                        joblib.dump(model, f'model/{model_name}.joblib')
                        st.experimental_rerun()
                elif st.session_state['是否训练']==2:
                        st.success("保存成功")


            elif train_or_load == "微调模型":
                models_dir = "model/"
                models_list = os.listdir(models_dir)
                model_to_load = st.selectbox("请选择一个模型：", models_list)
                model = joblib.load(f"{models_dir}{model_to_load}")
                if st.button("开始微调"):
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    st.write(f"{model_to_load} Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
                    st.write(f"{model_to_load} Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
                    st.write(f"{model_to_load} R-squared: {r2_score(y_test, y_pred)}")

                    save_model = st.button("保存微调后的模型")
                    if save_model:
                        model_name = st.text_input("请输入微调后的模型名称：",value=target+'--'+gen_model_name(model))
                        joblib.dump(model, f'model/{model_name}_tuned.joblib')
                        st.write(f"微调后的模型已保存为 'model/{model_name}_tuned.joblib'！")
    with tab2:

        import os
        import json
        import streamlit as st

        MODELS_PATH = st.session_state.get('block', '')  # 修改为您存放模型的实际路径


        # 获取指定路径下的所有模型
        def get_models_from_path(path=MODELS_PATH):
            models = [f for f in os.listdir(path) if f.endswith('.joblib')]
            return models


        # 获取指定路径下的所有模型
        available_models = get_models_from_path()


        # 根据模型名称筛选与目标相关的模型
        def filter_models_by_objective(objective, models):
            return [model for model in models if objective in model]


        # 目标值
        objectives = ['杨氏模量', '泊松比', '抗压强度', '内聚力', '内摩擦角', '孔隙压力']

        selected_models = {}

        # 尝试加载之前保存的模型选项索引
        try:
            with open(os.path.join(MODELS_PATH, "model_settings.json"), "r") as f:
                selected_models = json.load(f)
        except FileNotFoundError:
            pass

        # 对于每一个目标值，根据目标筛选模型并创建一个下拉菜单
        for obj in objectives:
            models_for_objective = filter_models_by_objective(obj, available_models)
            default_index = models_for_objective.index(
                selected_models.get(obj, models_for_objective[0])) if selected_models.get(obj) else 0
            selected_model = st.selectbox(f"{obj} 选择模型", models_for_objective, index=default_index)
            selected_models[obj] = selected_model

        # 当用户点击保存按钮时，保存选择的模型到本地的JSON文件
        if st.button("保存设置"):
            with open(os.path.join(MODELS_PATH, "model_settings.json"), "w") as f:
                json.dump(selected_models, f)
            st.success("设置已保存!")