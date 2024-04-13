
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import polars as pl
import streamlit as st

from src.wthor import LossCSVHandler

# losshandler = LossCSVHandler("app/resources/loss.csv")
df_id_name = pl.read_csv("app/resources/id_name.csv")

st.set_page_config(page_title="プレイヤーの統計値")

# ユーザー入力の取得
st.write(
    """
    # プレイヤーの統計値
    """
)
selected_player = st.selectbox('プレイヤー名を選択してください', options=df_id_name.get_column('PlayerName').sort())
selected_years = st.multiselect('集計対象の年を選択してください', options=range(2001, 2024), default=range(2014, 2024))
st.write('該当データがない場合はエラーになります')
if st.button('指標を計算'):
    # Retrieve the selected player's ID
    player_id = df_id_name.filter(pl.col('PlayerName') == selected_player).get_column('PlayerId')[0]

    # Initialize the LossCSVHandler for the selected player
    loss_handler = LossCSVHandler(f"app/resources/loss_playerId/loss_{player_id}.csv")
    
    # display dataframe
    df_show = loss_handler.create_basic_stats_df([player_id], selected_years)
    st.write(f"## {selected_player} さんの基本データ")
    st.dataframe(df_show)

    # Generate and display the plot
    st.write("## 10手ごとの合計石損")
    fig = loss_handler.plot_PlayerLossOver10Moves(player_id, selected_years, loss_column='PlayerTotalLossOver10Moves')
    st.pyplot(fig=fig)

    st.write("## 10手ごとの合計WLD損")
    fig = loss_handler.plot_PlayerLossOver10Moves(player_id, selected_years, loss_column='PlayerTotalWLDOver10Moves')
    st.pyplot(fig=fig)


    # 独自指標
    st.write("## 試験的な指標")
    st.write("### 10手ごとの合計sigmoid損")
    fig = loss_handler.plot_PlayerLossOver10Moves(player_id, selected_years, loss_column='PlayerSigmoidLossOver10Moves')
    st.pyplot(fig=fig)

    # 対戦履歴
    df_loss = loss_handler.get_loss_df()
    df_show = loss_handler.create_game_history_df(selected_player, selected_years)
    st.write("## 対戦履歴")
    st.dataframe(df_show)

    