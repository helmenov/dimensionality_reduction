import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

def PreCutoff(df):
    # 正規化
    ss = StandardScaler()
    # fit_transform関数は、fit関数（正規化するための前準備の計算）と
    # transform関数（準備された情報から正規化の変換処理を行う）の両方を行う
    df_st = pd.DataFrame(ss.fit_transform(df), index=df.index, columns=df.columns)

    # 25%点の絶対値もしくは75%点の絶対値のどちらかが、0.5以下
    df_cutoff = df.iloc[:,list(np.logical_not(np.any(np.abs(df_st.quantile([0.25,0.75])) < 0.5, axis=0)))]

    # 返すのは，正規化前のデータ
    return df_cutoff

def BasedByCorr(df: pd.DataFrame, corr_threshold=0.9) -> pd.DataFrame:
    # 相関係数の絶対値が>0.9ならマルチコ
    Adjacency = np.array(df.corr().abs() > corr_threshold).astype('int')
    while True:
        Adjacency_next = np.dot(Adjacency, Adjacency).astype('int')
        Adjacency_next[Adjacency_next != 0] = 1
        if(np.all(Adjacency_next == Adjacency)):
            break
        else:
            Adjacency = Adjacency_next
    multico_list = [np.where(Ai)[0].tolist() for Ai in Adjacency]

    ml = list()
    for m in multico_list:
        if len(m)>1:
            print(f'{df.columns[m]} are strongly correlated\n')
        ml.append(m[:1])
    extract_list = sorted(list(set(sum(ml,[]))))
    #extract_list = sorted(list(set(sum([m[:1] for m in multico_list], []))))

    # マルチコ危険な変数を排除したデータを作る
    df_corr = df.iloc[:, extract_list]

    return df_corr


def BasedByVIF(df: pd.DataFrame, vif_threshold=10) -> pd.DataFrame:
    vif = pd.DataFrame()

    # initialize vif
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif.index = df.columns

    # すべてのVIFが閾値を下回るまで、VIFの計算と列の除去を交互に繰り返します。
    # VIFの計算と列の除去を交互に繰り返す
    while True:
        df_vif = df[vif.index].copy()
        vif["VIF Factor"] = [
            variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])
        ]
        display(vif)
        if vif["VIF Factor"].max(axis=0) > vif_threshold:  # すべてのVIFがしきい値を下回るまで列を除去
            vif = vif.drop(vif["VIF Factor"].idxmax(axis=0), axis=0)
        else:
            break
    return df_vif
