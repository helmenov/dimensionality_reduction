import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy import stats as scistats
from sklearn.preprocessing import StandardScaler
import typing

def PreCutoff(df: pd.DataFrame)->pd.DataFrame:
    """十分に分散していない説明変数を切り捨てる

    Args:
        df (pd.DataFrame): 説明変数

    Returns:
        pd.DataFrame: カット済み説明変数
    """
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
    """相関係数が強すぎる列群は，マルチコの危険があるので，並び順の最も若い列に代表させる．

    Args:
        df (pd.DataFrame): 説明変数
        corr_threshold (float, optional): 相関係数のしきい値. Defaults to 0.9.

    Returns:
        pd.DataFrame: カット済み説明変数
    """
    ss = StandardScaler()
    df_st = pd.DataFrame(ss.fit_transform(df), index=df.index, columns=df.columns)
    # 相関係数の絶対値が>0.9ならマルチコ
    Adjacency = np.array(df_st.corr().abs() > corr_threshold).astype('int')
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
    """VIF(Variation Inflation Factor)の大きい説明変数を1つずつ取り除く

    Args:
        df (pd.DataFrame): 説明変数
        vif_threshold (int, optional): VIFしきい値. Defaults to 10.

    Returns:
        pd.DataFrame: カット済み説明変数
    """
    ss = StandardScaler()
    x = pd.DataFrame(ss.fit_transform(df), index=df.index, columns=df.columns)

    while True:
        print(x)
        x = add_constant(x)
        vif = pd.DataFrame()
        vif.index = x.columns
        df_vif = x[vif.index].copy()
        #vif["VIF Factor"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        vif_ = list()
        for i in range(df_vif.shape[1]):
            try:
                vif_i = variance_inflation_factor(df_vif.values,i)
            except RuntimeWarning: # `python -W error foo.py` catch
                print(f'{vif.index[i]} is huge vif')
                vif_i = 1e+7
            finally:
                vif_.append(vif_i)
        vif["VIF Factor"] = vif_

        vif = vif.drop('const',axis=0)
        vif_max = vif["VIF Factor"].max(axis=0)
        print(f'{vif_max=}')
        if vif_max > vif_threshold:  # すべてのVIFがしきい値を下回るまで列を除去
            vif = vif.drop(vif["VIF Factor"].idxmax(axis=0), axis=0)
            x = df_vif[vif.index].copy()
        else:
            break
    return df[vif.index].copy()

def codes4p(p_value:float)->str:
    """evaluated codes by p_value

    Args:
        p_value (float): probability to H0

    Returns:
        codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    """
    if p_value > 0.1: codes = ''
    elif p_value > 0.05: codes = '.'
    elif p_value > 0.01: codes = '*'
    elif p_value > 0.001: codes = '**'
    else: codes = '***'
    return codes

def F_test(a:pd.Series,b:pd.Series)->typing.Tuple[float,float,str]:
    """F検定

    Args:
        a (pd.Series): A群
        b (pd.Series): B群

    Returns:
        f_score, p_value, codes
    """
    # f_value
    u_a, u_b = np.var(a.values,ddof=1), np.var(b.values,ddof=1)
    n_a, n_b = a.size, b.size
    f_score = u_a/u_b
    #print(f'f_score=\n\t({u_a=:e})\n\t----------\n\t({u_b=:e})\n\t={f_value}')
    f_frozen = scistats.f.freeze(dfn=n_a-1, dfd=n_b-1)

    # left_side
    p_low = f_frozen.cdf(f_score)
    # right_side
    p_up = f_frozen.sf(f_score)
    # p_value
    p_value = min(p_low, p_up) * 2
    #
    codes = codes4p(p_value)

    return f_score, p_value, codes

def BasedBySignificance(df:pd.DataFrame, target_names=[1,-1], y=-1)->pd.DataFrame:
    """目的変数で分けた2群に有意差がない説明変数を削る

    Args:
        df (pd.DataFrame): 説明変数と目的変数をまとめたデータフレーム
        target_names (list, optional): 対象とする目的変数の水準2つ. Defaults to [1,-1].
        y (int, optional): dfにおける目的変数の列. Defaults to -1.

    Returns:
        pd.DataFrame: カット済み説明変数
    """
    #dfは[features,target]のDataFrame
    if isinstance(y,int):
        y = df.columns[y]
    features = df.drop(columns=y, axis=1)
    target = df[y]

    for v in features.columns:
        a = features[target==target_names[0]][v]
        b = features[target==target_names[1]][v]
        # a,bそれぞれの正規性(shapiro)
        assert scistats.shapiro(a).pvalue > 0.05
        assert scistats.shapiro(b).pvalue > 0.05
        # a,bの等分散性
        _,p_value,_ = ftest(a,b)
        if p_value > 0.05:
            p = scistats.ttest_ind(a,b,equal_var=True).pvalue
        else:
            p = scistats.ttest_ind(a,b,equal_var=False).pvalue
        if p > 0.05:
            features = features.drop(columns=v, axis=1)
    return features
