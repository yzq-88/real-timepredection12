
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from tool import *
st.session_state['平滑比例'] = 0.1
st.session_state['迭代次数'] = 1
st.set_page_config(layout="wide")
option=['文件输入', '声波数据处理', '声波平滑处理','计算处理']

if 'radio_value' not in st.session_state:
    st.session_state.radio_value = 0

with st.expander("1️⃣文件输入",expanded=True):
    st.write("📤上传测井数据")
    train_data_file = st.file_uploader("选择测井数据文件 (.txt)", type=["txt"])
    if train_data_file is not None:
        data=fx(train_data_file)
        st.write('处理后数据:',data.shape)
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=True)
        gb.configure_selection("single")
        gb.configure_grid_options(domLayout='autoHeight', paginationPageSize=20)
        grid_options = gb.build()
        response = AgGrid(
            data,
            gridOptions=grid_options,
            height=50,
            width='100%',
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,  # Set it to True to allow jsfunction to be used
            enable_enterprise_modules=True,
            license_key=None
        )
        st.session_state.radio_value = 1

with st.expander("2️⃣声波趋势线确定"):
    if st.session_state.radio_value==1:
        st.write('趋势线确定')
        col1, col2, col3,col4 = st.columns(4)
        b1=col1.number_input("B小数", value=-6.25, step=0.1)
        b2=col2.number_input("B指数", value=-5, step=1)
        A=col3.number_input("A", value=2.68, step=0.1)
        col4.latex(r"y_{\text{声波速度}} = 10^{A \cdot \text{井深} + B}")
        B=b1**b2
        st.session_state['A1']=B
        st.session_state['B1']=A
        col1, col2 ,col3= st.columns([1,3,1])
        with col2:
            nihe_plot(data, B, A)
            st.session_state.radio_value = 2
    else:
        st.warning("请按顺序执行")
with st.expander("3️⃣声波平滑处理"):
    if st.session_state.radio_value > 1:
        st.write('数据平滑')
        data = pd.read_excel("./temp_data/1.xlsx")
        col1, col2 = st.columns(2)
        smooth_fraction = col1.number_input("平滑比例", value=0.1, step=0.1)
        iterations = col2.number_input("迭代次数", value=1, step=2)
        st.session_state['平滑比例']=smooth_fraction
        st.session_state['迭代次数']=iterations
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            AC_smoother(data,smooth_fraction,iterations)
    else:
        st.warning("请按顺序执行")


with st.expander("4️⃣计算处理"):
    if st.session_state.radio_value > 1:
        st.write('岩石力学参数和孔隙压力计算')
        col1,col2=st.columns(2)
        data = pd.read_excel("./temp_data/1.xlsx")
        if st.button('开始处理'):
            data=process_pipeline(data)
            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_pagination()
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=True)
            gb.configure_selection("single")
            gb.configure_grid_options(domLayout='autoHeight')
            gb.configure_grid_options(domLayout='autoHeight', paginationPageSize=20)
            grid_options = gb.build()
            response = AgGrid(
                data,
                gridOptions=grid_options,
                height=50,
                width='100%',
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                allow_unsafe_jscode=True,  # Set it to True to allow jsfunction to be used
                enable_enterprise_modules=True,
                license_key=None
            )
            downd_result(data)
    else:
        st.warning("请按顺序执行")



#
#
#
