
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import polars as pl
import streamlit as st

from src.wthor import LossCSVHandler

# losshandler = LossCSVHandler("app/resources/loss.csv")
df_id_name = pl.read_csv("app/resources/id_name.csv")

st.set_page_config(page_title="ランキング")

# ユーザー入力の取得
st.write(
    """
    # プレイヤーランキング
    選択したプレイヤー内で、統計値のランキングを作成します。  
    選択数が多いと処理に時間がかかる場合があります。  
    """
)
selected_players = st.multiselect('プレイヤー名を選択してください', options=df_id_name.get_column('PlayerName').sort())
selected_years = st.multiselect('集計対象の年を選択してください', options=range(2001, 2024), default=range(2014, 2024))
st.write('該当データがない場合はエラーになります')
if st.button('指標を計算'):
    # Retrieve the selected player's ID
    player_ids = df_id_name.filter(pl.col('PlayerName').is_in(selected_players)).get_column('PlayerId').to_numpy()

    # Initialize the LossCSVHandler for the selected player
    loss_handler = LossCSVHandler("")
    df_loss = pl.concat([pl.read_csv(f"app/resources/loss_playerId/loss_{player_id}.csv") for player_id in player_ids])
    df_loss = df_loss.with_columns([
                pl.col('PlayerLoss').map_elements(loss_handler.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerLoss'),
                pl.col('OpponentLoss').map_elements(loss_handler.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('OpponentLoss'),
                pl.col('PlayerScore').map_elements(loss_handler.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerScore'),
                pl.col('PlayerIsWin').map_elements(loss_handler.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerIsWin'),
                pl.col('PlayerWLD').map_elements(loss_handler.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerWLD'),
            ])
    loss_handler.set_loss_df(df_loss)
    
    # display dataframe
    df_show = loss_handler.create_basic_stats_df(players=player_ids, years=selected_years, player_id_flg=1)
    df_show = df_show.join(df_id_name, how="left", on="PlayerId").select([
        pl.col(c).alias(c) for c in [
            'PlayerName', 'Color', '# Games', 'WinRate', 'WinRate@40', 'W@60,W@40', 'W@60,L@40', 'L@60,W@40', 'L@60,L@40', 'PerfectEndgame', 'EndgameLoss', 'EndgameSigmoidLoss',
        ]
    ]).filter(pl.col('PlayerName').is_in(selected_players))
    st.write(f"## 選択されたプレイヤーの手番別データ\n デフォルトは勝率順。項目をクリックで並び替え可能")
    st.dataframe(df_show.sort('WinRate', descending=True))


    