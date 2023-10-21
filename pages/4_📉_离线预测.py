import streamlit as st
import psycopg2
import random
import pandas as pd
import numpy as np
import os
import time
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from datetime import datetime
from tensorflow.keras.models import load_model
from attention import Attention
star_depth=1
current_depth=1
well_name='朝22X'
def read_excel_to_df(file_path):
    global star_depth
    global current_depth
    df = pd.read_excel(file_path)
    if star_depth==1 or star_depth<df['井深'].min():
        star_depth=df['井深'].min()
    if current_depth<star_depth:
        current_depth=star_depth
    return df

def simulate_api2(df):
    if current_depth<len(df)+1:
        features =  ["井名", '井深', '大勾载荷', '泵压', '机械钻速', '钻压', '转速', '排量', '密度', '粘度']
        df_row = df[df['井深'] == current_depth][features]
        # 更改列名
        df_row.columns = ["井名", '井深', '大勾载荷', '泵压', '机械钻速', '钻压', '转速', '排量', '密度', '粘度']
        return df_row, ''

    else:
        return "",'1'

def simulate_api(well_name):
    global current_depth
    data = {
        '井名': well_name,
        '井深': current_depth,
        '大勾载荷': random.uniform(100, 500),
        '泵压': random.uniform(0, 100),
        '机械钻速': random.uniform(0, 100),
        '钻压': random.uniform(0, 100),
        '转速': random.uniform(0, 100),
        '排量': random.uniform(0, 100),
        '密度': random.uniform(0, 100),
        "粘度": random.uniform(0, 100)
    }
    data=pd.DataFrame([data])
    return data,1

def real_api(well_name):
    data, ok = GetoneWellInfo("hua121_234X", "花121-234X", "fa04ca03-fa62-4f41-9065-b9faefa65e0a")
    return  data,ok

def insert_and_predict(conn, table_name, data, models, interval=0):
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {table_name} LIMIT 1')
    column_names = [description[0] for description in cur.description]
    columns = ', '.join(data.columns)
    values = ', '.join('%s' for _ in range(data.shape[1]))
    insert_query = f'INSERT INTO {table_name} ({columns}) VALUES ({values})'
    cur.executemany(insert_query, data.values.tolist())
    conn.commit()
    cur.execute(f'SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1')
    row = cur.fetchone()
    feature_names = ['大勾载荷', '泵压', '机械钻速', '钻压', '转速', '排量', '密度','粘度']
    features = [row[column_names.index(feature_name)] for feature_name in feature_names]
    features_2d = [features]
    for model_name, model in models.items():
        temp = pd.DataFrame(features_2d, columns= ['HKLD', 'SPP', 'ROP', 'WOB', 'RPM', 'FLOWIN', '密度', '粘度'])
        prediction = model.predict(temp)[0]
        prediction_scalar = float(prediction)

        cur.execute(f'UPDATE {table_name} SET "{model_name}" = %s WHERE id = %s', (prediction_scalar, row[0]))
    conn.commit()
    time.sleep(interval)

def get_max_well_depth(conn, table_name: str, well_name: str):
    global current_depth
    global star_depth
    cur = conn.cursor()
    cur.execute(f'SELECT 井深 FROM {table_name} WHERE "井名"=%s ORDER BY 井深 DESC Limit 1 ', (well_name,))
    try:
        temp_depth=cur.fetchall()[0][0]
        if star_depth<temp_depth:
            current_depth=temp_depth
        else:
            current_depth = star_depth
    except:
        print("error getMAX")
        current_depth=star_depth


def GetoneWellInfo(well_name_us: str, well_name_cn: str, token: str):
    global current_depth
    if token == "":
        token = "fa04ca03-fa62-4f41-9065-b9faefa65e0a"
    url1 = "http://10.76.204.222:10778/dataservices/services/%s.svc/VIEW?$filter=DEPTH ge %s and DEPTH le %s&access_token=%s" % (
        well_name_us, str(current_depth - 1), str(current_depth + 0), token)
    url2 = "http://10.76.204.222:10778/dataservices/services/zuanjingyexiangguan.svc/VIEW?$filter=WELL_NAME eq '%s'&access_token=%s" % (
        well_name_cn, token)
    res = {}
    default_num = -9999
    response1 = requests.get(url1)
    json_str1 = response1.content.decode('utf-8')
    data1 = json.loads(json_str1)  # 解析 JSON 数据
    if len(data1) == 0 or "error" in data1 or data1['value'] == []:
        return {}, "钻井参数为空。"
    WELLTIME = []
    DEPTH = []
    HKLD = []
    SPP = []
    VOP = []
    WOB = []
    RPM = []
    FLOWIN = []
    data = pd.DataFrame()
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    for item in data1['value']:
        WELLTIME.append(datetime.strptime(item['WELLTIME'], date_format).timestamp())
        DEPTH.append(item['DEPTH'])
        HKLD.append(item['HKLD'])
        SPP.append(item['SPP'])
        VOP.append(item['VOP'])
        WOB.append(item['WOB'])
        RPM.append(item['RPM'])
        FLOWIN.append(item['FLOWIN'])
    data["WELLTIME"] = WELLTIME
    data["DEPTH"] = DEPTH
    data["HKLD"] = HKLD
    data["SPP"] = SPP
    data["VOP"] = VOP
    data["WOB"] = WOB
    data["RPM"] = RPM
    data["FLOWIN"] = FLOWIN

    # res["WELLTIME"] = datetime.fromtimestamp(sum(WELLTIME) / len(data1['value'])).strftime(date_format)
    #     # res["DEPTH"] = sum(DEPTH) / len(data1['value'])
    # res["HKLD"] = sum(HKLD) / len(data1['value'])
    # res["SPP"] = sum(SPP) / len(data1['value'])
    # res["VOP"] = sum(VOP) / len(data1['value'])
    # res["WOB"] = sum(WOB) / len(data1['value'])
    # res["RPM"] = sum(RPM) / len(data1['value'])
    # res["FLOWIN"] = sum(FLOWIN) / len(data1['value'])
    print(url1)
    res["WELLTIME"] = datetime.fromtimestamp(data["WELLTIME"].mean()).strftime(date_format)
    res["DEPTH"] = current_depth
    res["HKLD"] = data["HKLD"].mean()
    res["SPP"] = data["SPP"].mean()
    res["VOP"] = data["VOP"].mean()
    res["WOB"] = data["WOB"].mean()
    res["RPM"] = data["RPM"].mean()
    res["FLOWIN"] = data["FLOWIN"].mean()

    response2 = requests.get(url2)
    json_str2 = response2.content.decode('utf-8')
    i = 0

    data2 = json.loads(json_str2)  # 解析 JSON 数据
    if len(data2["value"]) == 0:
        for v in ['WELL_ID', 'WELL_NAME', 'SAMPLING_TIME', 'SAMPLING_DEPTH', 'FLUID_DENSITY', 'FUNNEL_VISCOSITY', 'PV',
                  'READINGS_600RPM', 'READINGS_300RPM', 'READINGS_200RPM', 'READINGS_100RPM', 'YP',
                  'INITIAL_GEL_STRENGTH', 'GEL_STRENGTH_OF_10MINUTE']:
            res[v] = default_num
        return res, "钻井液参数为空"

    while i < len(data2['value']):
        if current_depth > data2['value'][i]['SAMPLING_DEPTH']:
            i += 1
        else:
            break
    if i >= len(data2["value"]):
        i = len(data2["value"]) - 1
    for v in ['WELL_ID', 'WELL_NAME', 'SAMPLING_TIME', 'SAMPLING_DEPTH', 'FLUID_DENSITY', 'FUNNEL_VISCOSITY', 'PV',
              'READINGS_600RPM', 'READINGS_300RPM', 'READINGS_200RPM', 'READINGS_100RPM', 'YP',
              'INITIAL_GEL_STRENGTH', 'GEL_STRENGTH_OF_10MINUTE']:
        res[v] = data2['value'][i][v]
    res=convert_dict_to_dataframe(res)
    res=res[["井名", '井深', '大勾载荷', '泵压', '机械钻速', '钻压', '转速', '排量', '密度', '粘度']]
    return res, ""

def GetStratDepth(well_name_us: str, token: str):
    global star_depth

    if token == "":
        token = "fa04ca03-fa62-4f41-9065-b9faefa65e0a"
    url1 = "http://10.76.204.222:10778/dataservices/services/%s.svc/VIEW?$filter=DEPTH ge %s and DEPTH le %s&access_token=%s" % (
        well_name_us, str(0), str(1000), token)
    response1 = requests.get(url1)
    json_str1 = response1.content.decode('utf-8')
    data1 = json.loads(json_str1)  # 解析 JSON 数据
    if len(data1) == 0 or "error" in data1:
        star_depth=1

    data = json.loads(json_str1)
    df = pd.read_json(json.dumps(data['value']), orient='records')
    star_depth = int(df["DEPTH"].min()+1)



def convert_dict_to_dataframe(data_dict):
    mapping_dict = {'WELLTIME': '记录时间', 'DEPTH': '井深', 'HKLD': '大勾载荷', 'SPP': '泵压', 'VOP': '机械钻速', 'WOB': '钻压', 'RPM': '转速', 'FLOWIN': '排量', 'WELL_ID': '井ID', 'WELL_NAME': '井名', 'SAMPLING_TIME': '采样时间', 'SAMPLING_DEPTH': '采样深度', 'FLUID_DENSITY': '密度', 'FUNNEL_VISCOSITY': '粘度', 'PV': '塑性粘度', 'READINGS_600RPM': 'f600', 'READINGS_300RPM': 'f300', 'READINGS_200RPM': 'f200', 'READINGS_100RPM': '12', 'YP': '动切力', 'INITIAL_GEL_STRENGTH': '初切力', 'GEL_STRENGTH_OF_10MINUTE': '终切力'}
    renamed_dict = {mapping_dict[key]: value for key, value in data_dict.items() if key in mapping_dict}
    df = pd.DataFrame([renamed_dict])
    return df

def load_models(model_names, path='models'):
    models = {name: joblib.load(os.path.join(path, model_names[name])) for name in model_names}
    return models


def get_well_data(conn, table_name, well_name):
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {table_name} WHERE "井名"=%s ORDER BY 井深', (well_name,))
    column_names = [desc[0] for desc in cur.description]
    return column_names, cur.fetchall()
def update_depth_time_annotations(fig):
    """更新图表的井深和时间批注"""
    global current_depth
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    annotations = [
        dict(
            x=0.3,  # 调整x位置以放置在左侧
            y=1.05,
            xref='paper',
            yref='paper',
            showarrow=False,
            text=f"当前井深: {current_depth}",
            font=dict(size=12)  # 设置字体大小为10
        ),
        dict(
            x=0.6,  # 调整x位置以放置在右侧
            y=1.05,
            xref='paper',
            yref='paper',
            showarrow=False,
            text=f"当前时间: {current_time_str}",
            font=dict(size=12)  # 设置字体大小为10
        )
    ]

    fig.update_layout(annotations=annotations)
    return fig

def dynamic_plot(column_names, data, well_name, model_names):

    depths = [row[2] for row in data]
    predictions = {model_name: [row[column_names.index(model_name)] for row in data] for model_name in model_names.keys()}
    num_models = len(model_names.keys())
    fig = make_subplots(rows=1, cols=num_models, shared_yaxes=True, subplot_titles=list(model_names.keys()))
    for i, model_name in enumerate(model_names.keys()):
        fig.add_trace(go.Scatter(x=predictions[model_name], y=depths, mode="lines", name=model_name), row=1, col=i+1)

    fig.update_layout(
                      yaxis=dict(title="井深", autorange="reversed", range=[6000, 0]),
                      xaxis=dict(title="预测值",tickangle=0),
                      grid=dict(rows=1),
                      showlegend=True)
    subplot_order = ["杨氏模量", "泊松比", "抗压强度", "内聚力", "内摩擦角", "孔隙压力"]
    for i in range(len(subplot_order)):
        fig.update_xaxes(title_text=subplot_order[i], tickangle=0, row=1, col=i + 1)
    fig = update_depth_time_annotations(fig)

    return fig
def dynamic_plot2(column_names, data, well_name, model_names):
    depths = [row[2] for row in data]
    predictions = {model_name: [row[column_names.index(model_name)] for row in data] for model_name in model_names.keys()}
    num_models = len(model_names.keys())
    fig2 = make_subplots(rows=1, cols=num_models, shared_yaxes=True, subplot_titles=list(model_names.keys()))
    for i, model_name in enumerate(model_names.keys()):
        fig2.add_trace(go.Scatter(x=predictions[model_name], y=depths, mode="lines", name=model_name), row=1, col=i+1)

    fig2.update_layout(
                      yaxis=dict(title="井深", autorange="reversed", range=[6000, 0]),
                      xaxis=dict(title="预测值",tickangle=0),
                      grid=dict(rows=1),
                      showlegend=True)
    subplot_order = ["杨氏模量", "泊松比", "抗压强度", "内聚力", "内摩擦角", "孔隙压力"]
    for i in range(len(subplot_order)):
        fig2.update_xaxes(title_text=subplot_order[i], tickangle=0, row=1, col=i + 1)
    fig2 = update_depth_time_annotations(fig2)

    return fig2
def load_model_settings(filename="model_settings.json", path=""):
    full_path=path+"/"+filename
    try:
        with open(full_path, "r") as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        print(f"文件 {full_path} 不存在!")
        return {}
    except json.JSONDecodeError:
        print(f"文件 {full_path} 格式错误!")
        return {}

def main():
    global current_depth
    global star_depth
    global well_name
    host = "localhost"
    database = "drilling_predication"
    user = "DR_LOGGING_CUP"
    password = "$Gy9eJ7!u2r5Y."
    table_name = "drilling"
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)

    col1,col2,col3=st.columns(3)
    with col1:
        directory_path = "./model"

        def get_subdirectories(directory_path):
            return [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        subdirs = get_subdirectories(directory_path)
        selected_dir = st.selectbox("请选择应用区块:", subdirs)
        if selected_dir!="":
            MODELS_PATH = "./model/" + selected_dir
            model_names = load_model_settings(path=MODELS_PATH)
            if len(model_names)>0:
                models = load_models(model_names, MODELS_PATH)
                st.session_state['block_seting']=1
            else:
                st.session_state['block_seting'] = 0
                st.warning("该区块下未设置模型")


    well_name = col2.text_input("井名：", "Well1")

    star_depth = col3.number_input("监测起始井深：", 0, 10000)
    current_depth=star_depth
    if st.session_state['block_seting']==1:
        uploaded_file = st.file_uploader(label='选取上传文件',
                                         type=['xlsx', 'xlsx'])
        # 如果有文件被上传
        if uploaded_file:
            # 使用Pandas读取CSV文件
            df = read_excel_to_df(uploaded_file)

        fig = dynamic_plot([], [], well_name, model_names)
        chart = st.plotly_chart(fig, use_container_width=True)

        fig2 = dynamic_plot2([], [], well_name, model_names)
        chart2 = st.plotly_chart(fig2, use_container_width=True)


        get_max_well_depth(conn, table_name, well_name)

        co1,co11,col2=st.columns([1,1,1])
        input_waiting=col2.empty()
        if co11.button("开始监控"):
            custom_objects = {'Attention': Attention}
            model_files = {
                "杨氏模量": "Ed_提前预测.h5",
                "泊松比": "Ud_提前预测.h5",
                "抗压强度": "St_提前预测.h5",
                "内聚力": "C_提前预测.h5",
                "内摩擦角": "fai_提前预测.h5",
                "孔隙压力": "孔隙压力_提前预测.h5"
            }
            loaded_models = {}
            for property_name, model_file in model_files.items():
                model = load_model(model_file, custom_objects=custom_objects)
                loaded_models[property_name] = model
            while True:
                data, ok = simulate_api2(df)
                if ok=='':
                    insert_and_predict(conn, table_name, data, models, interval=0)
#--------------------------------------------实时预测-------------------------------------------------
                    column_names, data = get_well_data(conn, table_name, well_name)
                    predictions = {model_name: [row[column_names.index(model_name)] for row in data] for model_name in model_names.keys()}
                    depths = [row[2] for row in data]
                    for i, model_name in enumerate(model_names.keys()):
                        fig.data[i].x = predictions[model_name]
                        fig.data[i].y = depths
                    new_min_depth, new_max_depth = min(depths), max(depths)+30
                    fig.update_yaxes(range=[new_max_depth+50, new_min_depth])
                    fig = update_depth_time_annotations(fig)
                    chart.plotly_chart(fig, use_container_width=True)
                    current_depth += 1
                    input_waiting.success("更新成功井深-"+str(current_depth))

 # --------------------------------------------提前预测-------------------------------------------------
                    if len(data)>80:
                        def predict_properties(data, column_names):
                            last_80_data = data[-80:]
                            last_80_data = pd.DataFrame(data=last_80_data, columns=column_names)

                            feature_names = ['大勾载荷', '泵压', '机械钻速', '钻压', '转速', '排量', '密度', '粘度']
                            last_80_data= last_80_data[feature_names]
                            last_80_data.columns=['HKLD', 'SPP', 'ROP', 'WOB', 'RPM', 'FLOWIN', '密度', '粘度']
                            last_80_data = np.array(last_80_data, dtype=np.float32)
                            input_data = np.expand_dims(last_80_data, axis=0)
                            # 使用提前加载的模型进行预测
                            prediction_results = {}
                            for property_name, model in loaded_models.items():
                                predictions = model.predict(input_data)
                                prediction_results[property_name] = predictions.flatten()

                            return prediction_results
                        predictions = predict_properties(data, column_names)
                        for i, model_name in enumerate(model_files.keys()):
                            fig2.data[i].x = predictions[model_name]
                            print(data[-1:][0][2])
                            fig2.data[i].y = [data[-1:][0][2]+i for i in range(30)]
                        chart2.plotly_chart(fig2, use_container_width=True)

                else:
                    input_waiting.success("暂未进尺"+str(current_depth))

                time.sleep(2)


if __name__ == "__main__":
    main()