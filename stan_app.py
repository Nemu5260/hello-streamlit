
#ライブラリの読み込み
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pystan
from scipy import stats
from tqdm import tqdm
import arviz
from sklearn.preprocessing import LabelEncoder

#タイトル
st.title("Stan実行アプリ")

# https://docs.streamlit.io/library/advanced-features/experimental-cache-primitives
@st.experimental_singleton
def button_states():
    return {"pressed": None}


def load_data(data,names=None,na_values=None,sep=None,skipinitialspace=False):
    data = pd.read_csv(data, names=names,na_values = na_values, 
                    sep=sep, skipinitialspace=skipinitialspace)
    return data

# @st.experimental_memo
def compile_model(model_code=None):
    model = pystan.StanModel(model_code=model_code)
    return model

def update_compile():
   st.write(st.session_state.model_code)

# 以下をサイドバーに表示
st.sidebar.markdown("### Upload csv file")
#ファイルアップロード
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files= False)

# csv読込方法の設定
st.sidebar.markdown("### Load data setting")
na_values = st.sidebar.text_input('Replace NAs(text)', "None")
# comment = st.sidebar.text_input('Comment(text)', "\t")
sep = st.sidebar.text_input('Sep(text)', ",")
skipinitialspace = st.sidebar.checkbox('Skip initial space(bool)')
# Load dataボタン
execute_loaddata = st.sidebar.button("Load Data")

# ボタンが押された状態を維持する
# https://qiita.com/pizatt775/items/6bf9e20e57bde078d459
is_pressed = button_states()
is_compiled = button_states()
if execute_loaddata:
    is_pressed.update({"pressed": True})

#ファイルがアップロードされ,Load dataボタンが押下されたら以下が実行される
if uploaded_files and is_pressed["pressed"]:

    # df = pd.read_csv(uploaded_files,names=column_names,na_values = na_values, 
    #             comment=comment, sep=sep, skipinitialspace=skipinitialspace)
    df = load_data(uploaded_files,na_values = na_values, 
                   sep=sep, skipinitialspace=skipinitialspace)
    df_columns = df.columns

    #データフレームを表示
    st.markdown("### Input Data")
    st.dataframe(df.style.highlight_max(axis=0))
    #matplotlibで可視化。X軸,Y軸を選択できる
    st.markdown("### EDA")
    #データフレームのカラムを選択オプションに設定する
    x = st.selectbox("X", df_columns)
    y = st.selectbox("Y", df_columns)
    #選択した変数を用いてmtplotlibで可視化
    fig = plt.figure(figsize= (12,8))
    plt.scatter(df[x],df[y])
    plt.xlabel(x,fontsize=18)
    plt.ylabel(y,fontsize=18)
    st.pyplot(fig)

    #seabornのペアプロットで可視化。複数の変数を選択できる。
    st.markdown("### Pair Plot")
    #データフレームのカラムを選択肢にする。複数選択
    item = st.multiselect("Columns", df_columns)
    #散布図の色分け基準を１つ選択する。カテゴリ変数を想定
    hue = st.selectbox("Standard Color", df_columns)
    
    #実行ボタン（なくてもよいが、その場合、処理を進めるまでエラー画面が表示されてしまう）
    execute_pairplot = st.button("Start Plot")
    #実行ボタンを押したら下記を表示
    if execute_pairplot:
            df_sns = df[item]
            df_sns["hue"] = df[hue]
            
            #streamlit上でseabornのペアプロットを表示させる
            fig = sns.pairplot(df_sns, hue="hue")
            st.pyplot(fig)


    st.markdown("### Modeling")
    st.markdown("#### Variable Setting")
    # カテゴリカル変数を選択
    ex_cate = st.multiselect("Choose Categirical variable（Multiple selection possible）", df_columns)


    #説明変数は複数選択式
    # ex = st.multiselect("Choose Explanatory variable（Multiple selection possible）", df_columns)
    ex = st.selectbox("Choose Explanatory variable（Multiple selection possible）", df_columns)

    #目的変数は一つ
    ob = st.selectbox("Choose Target variable", df_columns)

    st.markdown("#### Stan Code")
    # Stanコード入力欄
    st.text_area(
    "Enter Stan Code",
    height=400,
    # max_chars=850,
    key="model_code",
    on_change = update_compile)

    # compile_state = st.button("Compile")
    #実行ボタンを押したら下記を表示
    # if compile_state:
    #     # モデルのコンパイル
    #     compile_state = st.text('Compiling Model...')
    #     # model = pystan.StanModel(model_code=model_code)
    #     model = compile_model(model_code=st.session_state.model_code)
    #     compile_state.text("Finish Compiling Model.")
    #     is_compiled.update({"pressed": True})


    st.markdown("#### Sampling Setting")
    # Sampling設定
    iter=st.number_input("iter",10,20000,10)
    warmup=st.number_input("warmup",1,10000,1)
    chains=st.number_input("chains",1,4,1)
    thin=st.number_input("thin",1,2,1)
    seed=st.number_input('Seed(number)', 1,9999,1234)
    algorithm=st.selectbox('MCMC Algorithm', ("NUTS","HMC","Fixed_param"))

    #実行ボタンを押したら下記を表示
    execute_compiing = st.button("Start Compiling Model")
    if execute_compiing:
        # カテゴリ変数をラベルエンコーディング
        for elem in df_columns:
            if elem in ex_cate:
                le = LabelEncoder()
                le.fit(df[elem])
                df[elem] = le.transform(df[elem])
        # Stanデータ生成
        df = df[df[ob]!=2]
        standata = {
        'N': df.shape[0],
        'X': df[ex].values.astype(int),
        'Y': df[ob].values.astype(int),
        }

        st.markdown("#### Check Standata")
        st.write(standata)

        # standata.update(ex_dict)

        model_code = st.session_state.model_code
        st.markdown("#### Check StanCode")
        st.code(model_code,language="cpp")
        compile_state = st.text('Compiling Model...')
        model = compile_model(model_code=model_code)
        compile_state.text("Finish Compiling Model.")

        # Sampling開始        
        fit_state = st.text('Fitting Model...')

        fit = model.sampling(
            data=standata, 
            iter=iter, 
            warmup=warmup,
            # chains=chains,
            thin=thin, 
            n_jobs=-1, 
            seed=seed,
            algorithm=algorithm,
            verbose=True,
        )
        fit_state.text("Finish fitting")
        st.write(fit)
        fig = arviz.plot_trace(fit)
        st.pyplot(fig)


