#print("进入 PWV_APP.py 逻辑")

import os,sys
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
print('streamlit模块导入完成')

from matplotlib.font_manager import FontProperties

# 指定字体路径
app_path = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(app_path, 'SimHei.ttf')  # 结合 app 路径和字体文件名

# 确保字体路径是正确的
if not os.path.exists(font_path):
    raise FileNotFoundError(f"Font file not found: {font_path}")

# 设置字体属性
font_prop = FontProperties(fname=font_path)

# 设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()

# 用于显示中文和负号
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

os.environ["STREAMLIT_LOG_LEVEL"] = "error"  # 只显示错误信息



# 环境判断
if getattr(sys, 'frozen', False):
    # 在 EXE 环境中执行时，不再启动 Streamlit
    app_path = sys._MEIPASS  # 获取 EXE 解压后的临时路径
    print(f"Current Path: {app_path}")
else:
    # 在开发环境中执行
    app_path = os.path.dirname(os.path.abspath(__file__))

# 模型和 DLL 文件路径
model_path = os.path.join(app_path, 'xgboost_regressor_model.bin')
dll_path = os.path.join(app_path, 'xgboost', 'lib', 'xgboost.dll')
from xgboost import XGBRegressor
# 加载模型等其他应用逻辑
@st.cache_resource
def load_model(model_path):
    model = XGBRegressor()
    model.load_model(model_path)
    print('XGboost模型加载成功！')
    return model

model = load_model(model_path)

##正文

# 定义预测函数
def predict(features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]

# 设置应用程序的标题
st.title("PMV 预测程序")

# 创建选项卡来分隔四个子界面
tab1, tab2, tab3, tab4 , tab5, tab6 = st.tabs(["手动输入预测", "滑块预测","图形显示预测","图形显示预测(三维)","文件上传批量预测",'推测变量区间'])

# 特征重要性数据
feature_importances = {
    "环境温度（℃）": 0.523780,
    "MET代谢当量": 0.284665,
    "克罗Clo": 0.140117,
    "环境风速（m/s）": 0.036782,
    "环境相对湿度（%）": 0.014656,
}

# 特征顺序根据特征重要性排列
ordered_features = ["Air temperature", "Met",  "Clo", "Air velocity","Relative humidity"]
feature_mapping = {
    "Air temperature": "环境温度（℃）",
    "Met": "MET代谢当量",
    "Clo": "克罗Clo",
    "Air velocity": "环境风速（m/s）",
    "Relative humidity": "环境相对湿度（%）"
}

feature_mapping_reverse = {
    "环境温度（℃）":"Air temperature",
     "MET代谢当量":"Met",
    "克罗Clo":"Clo",
    "环境风速（m/s）":"Air velocity",
    "环境相对湿度（%）":"Relative humidity", 
}

# 第一个子界面：手动输入预测
with tab1:
    st.header("手动输入预测")
    
    # 按照重要性顺序排列输入框
    air_temp = st.number_input("环境温度（℃）", 0.60, 48.80, value=24.35, key="temp_input")
    met_val = st.number_input("MET代谢当量", 0.60, 6.80, value=1.21, key="met_input")
    clo_val = st.number_input("克罗Clo", 0.03, 2.87, value=0.70, key="clo_input")
    air_velocity = st.number_input("环境风速（m/s）", 0.00, 56.17, value=0.15, key="velocity_input")
    rel_humidity = st.number_input("环境相对湿度（%）", 0.50, 100.00, value=46.66, key="humidity_input")
    
    if st.button("预测 PMV"):
        features = [air_temp, met_val, air_velocity, clo_val, rel_humidity]
        prediction = predict(features)
        st.markdown(f"预测的 PMV 值为: <span style='color: skyblue; font-size: 20px; font-weight: bold'>{prediction:.2f}</span>", unsafe_allow_html=True)


    # 特征重要性图
    st.write(f"=="*43+"\n")
    st.subheader("各特征对 PMV 的影响")
    st.write(f"数值越大代表该特征对PMV的影响越大，反之越小")
    
    # 绘制柱状图
    fig, ax = plt.subplots()
    sns.barplot(x=list(feature_importances.values()), y=list(feature_importances.keys()),
                ax=ax, palette="Blues_d", hue=list(feature_importances.keys()))
    for i, v in enumerate(feature_importances.values()):
        ax.text(v, i, f"{v:.4f}", color='k', va='center')
    ax.set_xlabel("重要性数值")
    ax.set_title("PMV 预测的重要性")
    st.pyplot(fig)

# 第二个子界面：滑块预测
with tab2:
    st.header("滑块预测")
    
    # 按照重要性顺序排列滑块
    air_temp = st.slider("环境温度（℃）", 0.60, 48.80, value=24.35, key="temp_slider")
    met_val = st.slider("MET代谢当量", 0.60, 6.80, value=1.21, key="met_slider")
    clo_val = st.slider("克罗Clo", 0.03, 2.87, value=0.70, key="clo_slider")
    air_velocity = st.slider("环境风速（m/s）", 0.00, 56.17, value=0.15, key="velocity_slider")
    rel_humidity = st.slider("环境相对湿度（%）", 0.50, 100.00, value=46.66, key="humidity_slider")

    features = [air_temp, met_val, clo_val, air_velocity,  rel_humidity]
    prediction = predict(features)
    st.markdown(f"预测的 PMV 值为: <span style='color: skyblue; font-size: 20px; font-weight: bold'>{prediction:.2f}</span>", unsafe_allow_html=True)


# 第三个子界面：图形显示预测
with tab3:
    st.header("图形显示预测")

    # 选择菜单，选择一个变量
    selected_variable = feature_mapping_reverse[st.selectbox("选择一个变量", feature_mapping.values())]

    # 输入框手动输入其他变量的值
    st.subheader("手动输入其他变量的值")

    # 创建一个字典来保存其他变量的值
    fixed_values = {}

    for feature in ordered_features:
        if feature != selected_variable:
            # 添加输入框，仅对非选定变量进行输入
            if feature == "Air temperature":
                fixed_values[feature] = st.number_input("环境温度（℃）", 0.60, 48.80, value=24.35, key=feature)
            elif feature == "Met":
                fixed_values[feature] = st.number_input("MET代谢当量", 0.60, 6.80, value=1.21, key=feature)
            elif feature == "Clo":
                fixed_values[feature] = st.number_input("克罗Clo", 0.03, 2.87, value=0.70, key=feature)
            elif feature == "Air velocity":
                fixed_values[feature] = st.number_input("环境风速（m/s）", 0.00, 56.17, value=0.15, key=feature)
            elif feature == "Relative humidity":
                fixed_values[feature] = st.number_input('环境相对湿度（%）', 0.50, 100.00, value=46.66, key=feature)

    # 获取选择的变量的范围
    variable_range = {
        "Air temperature": (0.60, 48.80),
        "Met": (0.60, 6.80),
        "Clo": (0.03, 2.87),
        "Air velocity": (0.00, 56.17),
        "Relative humidity": (0.50, 100.00),
    }

    # 生成选定变量的取值范围
    var_min, var_max = variable_range[selected_variable]
    values = np.linspace(var_min, var_max, 100)

    # 计算每个取值下的 PMV
    pmv_values = []
    for value in values:
        # 更新选择变量的值
        fixed_values[selected_variable] = value

        # 使用更新后的变量值进行预测
        features = [fixed_values[feat] for feat in ordered_features]
        pmv = predict(features)
        pmv_values.append(pmv)

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=128)
    sns.scatterplot(x=values, y=pmv_values, marker="o", edgecolor='black', alpha=0.4, ax=ax)
    ax.set_xlabel(feature_mapping[selected_variable])
    ax.set_ylabel("预测 PMV")
    ax.set_title(f"不同 {selected_variable} 下预测的PMV变化")

    # 在图的左上角显示其他4个变量的取值
    ax.text(0.05, 0.95, "\n".join([f"{feature_mapping[k]}: {v}" for k, v in fixed_values.items() if k != selected_variable]),
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')

    # 隐藏其他边界线，只保留x轴和y轴的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    st.pyplot(fig)



with tab4:
    st.header("两个变量与PMV的三维关系图")
    # 选择两个变量
    selected_variable_1 = st.selectbox("选择一个变量", feature_mapping.values(), key="selectbox_1")
    selected_variable_1=feature_mapping_reverse[selected_variable_1]
    selected_variable_2 = st.selectbox("选择第二个变量", [feat for feat in feature_mapping.values() if feat != selected_variable_1], key="selectbox_2")
    selected_variable_2=feature_mapping_reverse[selected_variable_2]
    # 输入框手动输入其他三个变量的值
    st.subheader("手动输入其他变量的值")

    # 创建一个字典来保存其他变量的值
    fixed_values = {}

    for feature in ordered_features:
        if feature not in [selected_variable_1, selected_variable_2]:
            if feature == "Air temperature":
                fixed_values[feature] = st.number_input(feature_mapping[feature], 0.60, 48.80, value=24.35, key=f"{feature}_manual_2")
            elif feature == "Met":
                fixed_values[feature] = st.number_input(feature_mapping[feature], 0.60, 6.80, value=1.21, key=f"{feature}_manual_2")
            elif feature == "Clo":
                fixed_values[feature] = st.number_input(feature_mapping[feature], 0.03, 2.87, value=0.70, key=f"{feature}_manual_2")
            elif feature == "Air velocity":
                fixed_values[feature] = st.number_input(feature_mapping[feature], 0.00, 56.17, value=0.15, key=f"{feature}_manual_2")
            elif feature == "Relative humidity":
                fixed_values[feature] = st.number_input(feature_mapping[feature], 0.50, 100.00, value=46.66, key=f"{feature}_manual_2")

    # 获取选择的变量的范围
    variable_range = {
        "Air temperature": (0.60, 48.80),
        "Met": (0.60, 6.80),
        "Clo": (0.03, 2.87),
        "Air velocity": (0.00, 56.17),
        "Relative humidity": (0.50, 100.00),
    }

    # 生成选定变量的取值范围
    var_min_1, var_max_1 = variable_range[selected_variable_1]
    var_min_2, var_max_2 = variable_range[selected_variable_2]

    # 创建网格用于绘图
    values_1 = np.linspace(var_min_1, var_max_1, 20)
    values_2 = np.linspace(var_min_2, var_max_2, 20)
    X, Y = np.meshgrid(values_1, values_2)

    # 计算每个网格点的 PMV
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fixed_values[selected_variable_1] = X[i, j]
            fixed_values[selected_variable_2] = Y[i, j]
            
            # 使用更新后的变量值进行预测
            features = [fixed_values[feat] for feat in ordered_features]
            Z[i, j] = predict(features)

    # 绘制三维散点图
    fig = plt.figure(figsize=(9, 7), dpi=128)  # 调整图表尺寸
    ax = fig.add_subplot(111, projection='3d')

    # 根据PMV值来为每个点着色
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o', alpha=0.7)

    # 设置轴标签
    ax.set_xlabel(feature_mapping[selected_variable_1])
    ax.set_ylabel(feature_mapping[selected_variable_2])
    ax.set_zlabel("预测 PMV")
    ax.set_title(f"不同{feature_mapping[selected_variable_1]} 和 {feature_mapping[selected_variable_2]} 下的PMV预测值")

    # 显示颜色条
    fig.colorbar(sc, ax=ax, label='PMV')

    # 显示其他三个固定变量的取值
    remaining_vars = [var for var in ordered_features if var not in [selected_variable_1, selected_variable_2]]
    text = "\n".join([f"{feature_mapping[var]}: {fixed_values[var]}" for var in remaining_vars])

    ax.text2D(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    st.pyplot(fig)

required_columns = [ "Air temperature","Met", "Clo", "Air velocity (m/s)", "Relative humidity (%)", ]
with tab5:
    st.header("文件上传批量预测")
    uploaded_file = st.file_uploader("上传一个 Excel 或 CSV 文件", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # 检查是否包含所有需要的列
        if all(column in df.columns for column in required_columns):
            X = df[required_columns]
            predictions = model.predict(X)
            df['Predicted PMV'] = predictions
            df.to_csv('pred_result.csv', index=False)
            st.write("预测结果已保存到 pred_result.csv")
            st.write(df.head())
        else:
            st.write("上传的文件缺少必要的列")

    # 案例数据表格
    st.write(f"=="*43+"\n")
    st.subheader("案例数据格式参考")
    st.write(f"上传文件必须包含这5列变量，且表头名称要和如下示例一模一样：")
    example_data = {
        "Clo": [0.57, 0.57, 0.57, 0.57, 0.57],
        "Met": [1.0, 1.1, 1.1, 1.0, 1.0],
        "Air temperature": [24.3, 25.7, 24.6, 26.4, 25.0],
        "Relative humidity (%)": [36.8, 33.1, 34.9, 31.7, 33.3],
        "Air velocity (m/s)": [0.27, 0.09, 0.06, 0.13, 0.07],
    }
    example_df = pd.DataFrame(example_data)[required_columns]
    st.write(example_df)

def find_x1_ranges(x1_values, y_pred, y_min, y_max, selected_variable_3,show_fig=False):
    valid_ranges = []
    current_range = []

    for x1, y in zip(x1_values, y_pred):
        if y_min <= y <= y_max:
            if not current_range:
                current_range = [x1, x1]  # 开始新的区间
            else:
                current_range[1] = x1  # 更新当前区间的结束值
        else:
            if current_range:
                valid_ranges.append(tuple(current_range))  # 结束当前区间并保存
                current_range = []

    if current_range:
        valid_ranges.append(tuple(current_range))

    if show_fig:
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=128)
        ax.scatter(x1_values, y_pred, color='skyblue', edgecolor='black', alpha=0.5, label='Data Points')

        for start, end in valid_ranges:
            plt.fill_between(x1_values, y_min, y_max, where=(x1_values >= start) & (x1_values <= end),
                             color='gold', alpha=0.2, label='Target Range' if start == valid_ranges[0][0] else "")

            mid_point = (start + end) / 2
            ax.text(mid_point, y_min, f'[{start:.2f}, {end:.2f}]',
                    color='black', fontsize=10, ha='center', va='center', alpha=1)

        plt.axhline(y_min, color='r', linestyle='-.', linewidth=1)
        plt.axhline(y_max, color='r', linestyle='-.', linewidth=1)

        for start, end in valid_ranges:
            ax.axvline(start, color='b', linestyle='--', linewidth=1)
            ax.axvline(end, color='b', linestyle='--', linewidth=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('PMV区间')
        ax.set_xlabel(f'{feature_mapping[selected_variable_3]}')
        ax.set_ylabel('PMV')
        st.pyplot(fig)

    return valid_ranges

# 增加一个新子界面
with tab6:
    st.header("变量与PMV范围计算和推测")

    # 用户选择一个变量
    selected_variable_3 = st.selectbox("选择一个需要推测区间的变量", feature_mapping.values(), key="selectbox_3")
    selected_variable_3=feature_mapping_reverse[selected_variable_3]

    # 用户输入 PMV 的目标范围
    col1, col2 = st.columns(2)
    with col1:
        y_min = st.number_input("PMV的最小值", -5.0, 5.0, value=-0.5, key="PMV最小值")
    with col2:
        y_max = st.number_input("PMV的最大值", -5.0, 5.0, value=0.5, key="PMV最大值")

    # 输入其他变量的固定值
    st.subheader("手动输入其他变量的值")
    fixed_values = {}
    
    for feature in ordered_features:
        if feature != selected_variable_3:
            # 添加输入框，仅对非选定变量进行输入
            if feature == "Air temperature":
                fixed_values[feature] = st.number_input("环境温度（℃）", 0.60, 48.80, value=24.35, key=f"{feature}_2")
            elif feature == "Met":
                fixed_values[feature] = st.number_input("MET代谢当量", 0.60, 6.80, value=1.21, key=f"{feature}_2")
            elif feature == "Clo":
                fixed_values[feature] = st.number_input("克罗Clo", 0.03, 2.87, value=0.70, key=f"{feature}_2")
            elif feature == "Air velocity":
                fixed_values[feature] = st.number_input("环境风速（m/s）", 0.00, 56.17, value=0.15, key=f"{feature}_2")
            elif feature == "Relative humidity":
                fixed_values[feature] = st.number_input('环境相对湿度（%）', 0.50, 100.00, value=46.66, key=f"{feature}_2")


    # 获取选择的变量的范围
    variable_range = {
        "Air temperature": (0.60, 48.80),
        "Met": (0.60, 6.80),
        "Clo": (0.03, 2.87),
        "Air velocity": (0.00, 56.17),
        "Relative humidity": (0.50, 100.00),
    }

    # 生成选定变量的取值范围
    var_min, var_max = variable_range[selected_variable_3]
    values = np.linspace(var_min, var_max, 100)

    # 计算每个取值下的 PMV
    pmv_values = []
    for value in values:
        # 更新选择变量的值
        fixed_values[selected_variable_3] = value

        # 使用更新后的变量值进行预测
        features = [fixed_values[feat] for feat in ordered_features]
        pmv = predict(features)
        pmv_values.append(pmv)

    # 使用 find_x1_ranges 函数计算可能的取值区间
    valid_ranges = find_x1_ranges(values, pmv_values, y_min, y_max, selected_variable_3 ,show_fig=True)

    # 输出可能的取值区间
    if valid_ranges:
        range_text = ', '.join([f"[{start:.2f}, {end:.2f}]" for start, end in valid_ranges])
        st.markdown(f"**{feature_mapping[selected_variable_3]}** 变量的取值范围可能是: <span style='color: skyblue; font-size: 20px; font-weight: bold'>{range_text}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**{feature_mapping[selected_variable_3]}** 变量没有满足条件的取值范围。")
