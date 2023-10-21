import streamlit as st
import yaml
from yaml.loader import SafeLoader



import streamlit_authenticator as stauth

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('登录', 'main')
if authentication_status:
    authenticator.logout('退出', 'main')
    st.write(f'欢迎，{name}')
    st.success('欢迎进入')
elif authentication_status is False:
    st.error('用户名或密码不正确')
elif authentication_status is None:
    st.warning('请输入您的用户名和密码')
st.markdown(
    """
    <style>
        .stApp::before {
            content: "";
            display: block;
            padding-top: 50.25%;  /* 基于 16:9 宽高比的逆比例: 9/16 = 0.5625 */
            background: url("https://picss.sunbangyan.cn/2023/08/16/o4x2hb.jpg") center/cover no-repeat;
        }
    </style>
    """,
    unsafe_allow_html=True
)