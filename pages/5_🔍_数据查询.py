import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import psycopg2
st.set_page_config(layout="wide")
def database_query():
    # 连接到 Postgres 数据库
    host = "localhost"
    database = "drilling_predication"
    user = "DR_LOGGING_CUP"
    password = "$Gy9eJ7!u2r5Y."
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)

    # 获取井号列表
    df = pd.read_sql_query("SELECT DISTINCT \"井名\" FROM drilling", conn)
    well_names = df['井名'].tolist()
    selected_well = st.selectbox("选择一个井号", well_names)

    if st.button("执行查询"):
        if selected_well:
            # 执行按井查询并按“井深”排序
            query = f"SELECT * FROM drilling WHERE \"井名\" = '{selected_well}' ORDER BY \"井深\" ASC"
            df = pd.read_sql_query(query, conn)

            if not df.empty:
                # 使用streamlit-aggrid创建可筛选、翻页的表格
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=True)
                gb.configure_selection("single")
                gb.configure_grid_options(domLayout='autoHeight',paginationPageSize=20)
                grid_options = gb.build()

                response = AgGrid(
                    df,
                    gridOptions=grid_options,
                    # height=40,
                    width='100%',
                    fit_columns_on_grid_load=True,
                    update_mode=GridUpdateMode.NO_UPDATE,
                    enable_enterprise_modules=True,
                    enable_column_header=False,
                license_key=None
                )

                # 数据导出按钮
                if st.button("导出数据"):
                    df.to_csv("exported_data.csv", index=False)
                    st.success("数据已成功导出到 exported_data.csv")
            else:
                st.write("数据库中没有相关井号的数据")
        else:
            st.write("请选择一个井号")

if __name__ == "__main__":
    database_query()