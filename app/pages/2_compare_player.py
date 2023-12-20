import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go


# 文字列を数値リストに変換
def convert_string_to_list(string):
    return np.array([int(x.strip()) for x in string.strip('[]\n').split(' ') if x.strip().isdigit()])


def load_and_process_data(year):
    file_path = f"app/resources/addloss_wthor_{year}.csv"
    df_year = pl.read_csv(file_path)
    df_year = df_year.with_columns([
        pl.col('loss_black').map_elements(convert_string_to_list).alias('loss_black'),
        pl.col('loss_white').map_elements(convert_string_to_list).alias('loss_white'),
        pl.lit(year).alias('year')
    ])
    return df_year


def load_and_combine_csv(years):
    all_year_df = None
    for year in years:
        try:
            df_year = load_and_process_data(year)
            if all_year_df is None:
                all_year_df = df_year
            else:
                all_year_df = pl.concat([all_year_df, df_year])
        except FileNotFoundError:
            st.error(f'app/resources/lv18/addloss_wthor_{year}.csv が見つかりません。')

    return all_year_df


def pad_list_to_length_60(lst):
    """リストを長さ60に拡張する。必要に応じて0で埋める"""
    return np.array(lst.tolist() + [0] * (60 - len(lst)))


def make_df_id_name(df):
    df_black = df.select([
        pl.col('blackPlayerId').alias('PlayerId'),
        pl.col('blackPlayerName').alias('PlayerName')
    ]).unique()
    df_white = df.select([
        pl.col('whitePlayerId').alias('PlayerId'),
        pl.col('whitePlayerName').alias('PlayerName')
    ]).unique()
    return pl.concat([df_black, df_white]).unique()


def make_df_loss(df, df_id_name):
    # Black プレイヤー用の計算
    df_loss_black = df.filter(pl.col('blackPlayerId').is_in(df_id_name['PlayerId']))
    df_loss_black = df_loss_black.with_columns([
        pl.col('blackPlayerId').alias('PlayerId'),
        pl.col('whitePlayerId').alias('OpponentId'),
        pl.col('tournamentId'),
        pl.lit(0).alias('Color'),  # Black = 0
        pl.col('blackScore').alias('Score'),
        pl.col('blackTheoreticalScore').alias('TheoreticalScore'),
        pl.col('loss_black').alias('Player_Loss'),
        pl.col('loss_white').alias('Opponent_Loss'),
        pl.col('transcript'),
        pl.col('year')
    ])

    # White プレイヤー用の計算
    df_loss_white = df.filter(pl.col('whitePlayerId').is_in(df_id_name['PlayerId']))
    df_loss_white = df_loss_white.with_columns([
        pl.col('whitePlayerId').alias('PlayerId'),
        pl.col('blackPlayerId').alias('OpponentId'),
        pl.col('tournamentId'),
        pl.lit(1).alias('Color'),  # White = 1
        (64 - pl.col('blackScore')).alias('Score'),
        (64 - pl.col('blackTheoreticalScore')).alias('TheoreticalScore'),
        pl.col('loss_white').alias('Player_Loss'),
        pl.col('loss_black').alias('Opponent_Loss'),
        pl.col('transcript'),
        pl.col('year')
    ])

    # 両方のデータを結合
    df_loss = pl.concat([df_loss_black, df_loss_white])

    df_loss = df_loss.with_columns([
        pl.col('Player_Loss').map_elements(lambda x: pad_list_to_length_60(x)).alias('Player_Loss'),
        pl.col('Opponent_Loss').map_elements(lambda x: pad_list_to_length_60(x)).alias('Opponent_Loss')
    ])

    # Sum_Player_Loss と Sum_Player_Loss_24Empty の計算
    df_loss = df_loss.with_columns([
        pl.col('Player_Loss').map_elements(lambda x: sum(x)).alias('Sum_Player_Loss'),
        pl.col('Player_Loss').map_elements(lambda x: sum(x[-24:])).alias('Sum_Player_Loss_24Empty')
    ])

    return df_loss


def calculate_result(score):
    if score > 32:
        return 1
    elif score == 32:
        return 0.5
    else:
        return 0


def calculate_average_sum_player_loss_with_name(df_loss, df_id_name, count_min=10):
    # 1. 勝利と理論上の勝利の計算
    df_loss = df_loss.with_columns([
        pl.col('Score').map_elements(calculate_result).alias('Win'),
        pl.col('TheoreticalScore').map_elements(calculate_result).alias('Win Theoretical')
    ])

    # 2. プレイヤーごとの集計
    grouped = df_loss.group_by('PlayerId').agg([
        pl.col('Sum_Player_Loss').mean().alias('Average_Sum_Player_Loss'),
        pl.col('Sum_Player_Loss').count().alias('Record_Count'),
        pl.col('Sum_Player_Loss_24Empty').mean().alias('Average_Sum_Player_Loss_24Empty'),
        pl.col('Win').mean().alias('Win_Rate'),
        pl.col('Win Theoretical').mean().alias('Win_Rate_Theoretical')
    ])

    # 3. 'Record_Count' が count_min より大きいレコードのみを対象にする
    filtered_grouped = grouped.filter(pl.col('Record_Count') > count_min)

    # 4. df_id_name で PlayerId ごとに最初の PlayerName を取得し、df_result とマージ
    df_id_name_unique = df_id_name.unique(subset=['PlayerId'])
    df_result = filtered_grouped.join(df_id_name_unique, on='PlayerId')

    # 5. 'Average_Sum_Player_Loss' の小さい順にソート
    # df_result = df_result.sort(['Average_Sum_Player_Loss', 'Record_Count'], descending=[False, True])

    # 6. 数値の丸め処理
    df_result = df_result.with_columns([
        pl.col('Average_Sum_Player_Loss').round(2).alias('Average_Sum_Player_Loss'),
        pl.col('Average_Sum_Player_Loss_24Empty').round(2).alias('Average_Sum_Player_Loss_24Empty'),
        (pl.col('Win_Rate') * 100).round(1).alias('Win_Rate'),
        (pl.col('Win_Rate_Theoretical') * 100).round(1).alias('Win_Rate_Theoretical')
    ])

    # 7. 追い上げ勝ちと追い上げ負けの計算
    df_loss = df_loss.with_columns([
        (pl.col('Win') - pl.col('Win Theoretical')).clip(0).alias('Come From Behind W'),
        (pl.col('Win Theoretical') - pl.col('Win')).clip(0).alias('Come From Behind L')
    ])
    come_from_behind_grouped = df_loss.group_by('PlayerId').agg([
        pl.sum('Come From Behind W').alias('Come From Behind W'),
        pl.sum('Come From Behind L').alias('Come From Behind L')
    ])
    df_result = df_result.join(come_from_behind_grouped, on='PlayerId')
    df_result = df_result.with_columns([
        (pl.col('Come From Behind W') / pl.col('Record_Count') * 100).round(1).alias('Come-From-Behind Win (%)'),
        (pl.col('Come From Behind L') / pl.col('Record_Count') * 100).round(1).alias('Come-From-Behind Lose (%)')
    ])

    # 8. 最終的なデータフレームの整形とカラム名の変更
    df_rank_sorted = df_result.select([
        pl.col('PlayerId').alias('Player Id'),
        pl.col('PlayerName').alias('Player Name'),
        pl.col('Record_Count').alias('#Games'),
        pl.col('Average_Sum_Player_Loss').alias('Average Total Loss'),
        pl.col('Average_Sum_Player_Loss_24Empty').alias('Average 24 Empty Loss'),
        pl.col('Win_Rate').alias('Win Rate (%)'),
        pl.col('Win_Rate_Theoretical').alias('Win Rate @36 (%)'),
        pl.col('Come-From-Behind Win (%)'),
        pl.col('Come-From-Behind Lose (%)')
    ])

    df_rank_sorted = df_rank_sorted.sort(['Average Total Loss', '#Games'], descending=[False, True])

    return df_rank_sorted


# ランキングに基づいてグループを作成する関数
def get_player_ids_from_ranking(df_rank_sorted, start_rank, end_rank):
    if start_rank <= 0 or end_rank > len(df_rank_sorted):
        raise ValueError("指定されたランキング範囲が不正です。")
    # スライスして特定の列を選択し、リストに変換
    player_ids_list = df_rank_sorted.slice(start_rank - 1, end_rank - (start_rank - 1)) \
                                    .select('Player Id') \
                                    .get_column('Player Id') \
                                    .to_list()
    return player_ids_list


# ボタンが押されたときの処理
def on_button_clicked():
    st.session_state['combined_df'] = load_and_combine_csv(selected_years)
    st.session_state['df_id_name'] = make_df_id_name(st.session_state['combined_df'])
    st.session_state['df_loss'] = make_df_loss(st.session_state['combined_df'], st.session_state['df_id_name'])
    st.session_state['df_rank_sorted'] = calculate_average_sum_player_loss_with_name(st.session_state['df_loss'], st.session_state['df_id_name'], count_min=10)
    st.session_state['top10_playerids'] = get_player_ids_from_ranking(st.session_state['df_rank_sorted'], 1, 10)


# 6手ごとにセグメント化してlossを計算
def calculate_player_loss_sixmoves(df):
    # 6手ごとにセグメントを分け、各セグメント内の平均値の2倍を計算
    def segment_average(x):
        return [2 * np.mean(x[i:i + 6]) for i in range(0, len(x), 6)]

    df = df.with_columns(
        pl.col('Player_Loss').map_elements(segment_average).alias('Player_Loss_Sixmoves')
    )
    return df


# 1つのGroupの比較用関数
def plot_loss_sixmoves_boxplot(df_, player_id_list1):
    # Polarsでデータフレームのコピーを作成
    df = df_.clone()

    # 6手ごとの損失を計算
    df = calculate_player_loss_sixmoves(df)

    plt.figure(figsize=(15, 6))

    # グループごとに異なる色を割り当てるためのカラーマップ
    group_colors = ['red']

    # グループ1のデータを集計してプロット
    group1_array = df.filter(pl.col('PlayerId').is_in(player_id_list1)).get_column('Player_Loss_Sixmoves').to_numpy()
    group1_array = np.array([row for row in group1_array])

    # 各セグメントの位置を決定
    positions = np.arange(1, len(group1_array[0]) + 1)

    # ボックスプロットの描画
    plt.boxplot(group1_array, positions=positions, widths=0.3, patch_artist=True, showfliers=False, boxprops=dict(facecolor="white", color=group_colors[0]), medianprops=dict(color='black'))

    # 平均値のプロット
    plt.scatter(positions, np.mean(group1_array, axis=0), color=group_colors[0], edgecolor='black', zorder=3, label=f'Group 1 (n={len(group1_array)})')

    # プロットのカスタマイズ
    plt.title("Loss Values Boxplot and Average")
    plt.xlabel("Game Segment")
    plt.ylabel("Loss Value / Move")
    plt.xticks(range(1, 11), [f"{1+6*i}~{6*(i+1)}" for i in range(10)])
    plt.grid(True)
    plt.legend()
    return plt


# 2つのGroupの比較用関数
def plot_group_loss_sixmoves_boxplot(df_, player_id_list1, player_id_list2):
    df = df_.clone()
    df = calculate_player_loss_sixmoves(df)
    plt.figure(figsize=(15, 6))

    # グループごとに異なる色を割り当てるためのカラーマップ
    group_colors = ['red', 'blue']

    # グループ1のデータを集計してプロット
    group1_array = df.filter(pl.col('PlayerId').is_in(player_id_list1)).get_column('Player_Loss_Sixmoves').to_numpy()
    group1_array = np.array([row for row in group1_array])
    plt.boxplot(group1_array, positions=np.arange(1, len(group1_array[0]) + 1) - 0.15, widths=0.3, patch_artist=True, showfliers=False, boxprops=dict(facecolor="white", color=group_colors[0]), medianprops=dict(color='black'))
    plt.scatter(np.arange(1, len(group1_array[0]) + 1) - 0.15, np.mean(group1_array, axis=0), color=group_colors[0], edgecolor='black', zorder=3, label=f'Group 1 (n={len(group1_array)})')

    # グループ2のデータを集計してプロット
    group2_array = df.filter(pl.col('PlayerId').is_in(player_id_list2)).get_column('Player_Loss_Sixmoves').to_numpy()
    group2_array = np.array([row for row in group2_array])
    plt.boxplot(group2_array, positions=np.arange(1, len(group2_array[0]) + 1) + 0.15, widths=0.3, patch_artist=True, showfliers=False, boxprops=dict(facecolor="white", color=group_colors[1]), medianprops=dict(color='black'))
    plt.scatter(np.arange(1, len(group2_array[0]) + 1) + 0.15, np.mean(group2_array, axis=0), color=group_colors[1], edgecolor='black', zorder=3, label=f'Group 2 (n={len(group2_array)})')

    # プロットのカスタマイズ
    plt.title("Loss Values Boxplot and Average")
    plt.xlabel("Game Segment")
    plt.ylabel("Loss Value / Move")
    plt.xticks(range(1, 11), [f"{1+6*i}~{6*(i+1)}" for i in range(10)])
    plt.grid(True)
    plt.legend()
    return plt


def scatter_group_winrate(df1, df2):
    # 正方形のプロット領域を設定
    plt.figure(figsize=(6, 6))

    # グループごとに異なる色を割り当てるためのカラーマップ
    group_colors = ['red', 'blue']

    # 散布図をプロット
    plt.scatter(df1["Win Rate @36 (%)"], df1["Win Rate (%)"], color=group_colors[0], label="Group 1")
    plt.scatter(df2["Win Rate @36 (%)"], df2["Win Rate (%)"], color=group_colors[1], label="Group 2")

    # ラベルとタイトルを設定
    plt.xlabel("Win Rate @36 (%)")
    plt.ylabel("Win Rate (%)")
    plt.title("Win Rate Distribution")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    # 対角線（y=x）を点線で描画
    plt.plot([0, 100], [0, 100], linestyle='--', color='gray')

    # 凡例を表示
    plt.legend()

    return plt


def scatter_group_winrate_interactive(df1_, df2_):
    # Polarsでデータフレームのコピーを作成
    df1 = df1_.clone()
    df2 = df2_.clone()

    # df1とdf2にグループ識別列を追加
    df1 = df1.with_columns(pl.lit('Group 1').alias('Group'))
    df2 = df2.with_columns(pl.lit('Group 2').alias('Group'))

    # Polarsでデータフレームを結合
    df_combined = pl.concat([df1, df2])

    # Plotlyで散布図を作成
    fig = go.Figure()

    # Group 1 のデータを追加
    fig.add_trace(go.Scatter(x=df_combined.filter(pl.col('Group') == 'Group 1')['Win Rate @36 (%)'],
                            y=df_combined.filter(pl.col('Group') == 'Group 1')['Win Rate (%)'],
                            mode='markers', name='Group 1',
                            marker=dict(color='red'),
                            hoverinfo='text',
                            text=df_combined.filter(pl.col('Group') == 'Group 1')['Player Name']))

    # Group 2 のデータを追加
    fig.add_trace(go.Scatter(x=df_combined.filter(pl.col('Group') == 'Group 2')['Win Rate @36 (%)'],
                            y=df_combined.filter(pl.col('Group') == 'Group 2')['Win Rate (%)'],
                            mode='markers', name='Group 2',
                            marker=dict(color='blue'),
                            hoverinfo='text',
                            text=df_combined.filter(pl.col('Group') == 'Group 2')['Player Name']))

    # タイトルと軸ラベルを設定
    fig.update_layout(title="Win Rate Distribution",
                    xaxis_title="Win Rate @36 (%)",
                    yaxis_title="Win Rate (%)",
                    width=600,
                    height=600,
                    xaxis=dict(
                        scaleanchor="y",
                        scaleratio=1,
                        autorange=False,
                        range=[0, 100]
                    ),
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1,
                        autorange=False,
                        range=[0, 100]
                    ),
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',  # 透明な背景色
                    xaxis_showgrid=False,  # グリッドライン非表示
                    yaxis_showgrid=False)

    # 対角線（y=x）を点線で描画
    fig.add_shape(type="line",
                x0=0, y0=0, x1=100, y1=100,
                line=dict(color="gray", width=2, dash="dot"))

    return fig


# ユーザー入力の取得
st.write(
    """
    # プレイヤーの石損指標の計算
    """
)
selected_years = st.multiselect('集計対象の年を選択してください', options=range(2001, 2024), default=range(2019, 2024))

# ボタンを押すとCSVファイルを結合
if st.button('指定した年で計算', on_click=on_button_clicked):
    pass

# ランキングが計算されている場合のみグループ選択UIを表示
if 'df_rank_sorted' in st.session_state:
    # 結果を表示
    st.write(
        """
        ## 全体集計

        ### 平均石損ランキング
        - 集計対象: 期間内に10局以上の対局データがあるプレイヤー
        - 集計項目
            - #Games: 対局データ数
            - Average Total Loss: 試合中にAIの評価と比べてどれだけ石損したか
            - Average 24 Empty Loss: 最後の24手でどれだけ石損したか
            - Win Rate (%): 勝率
            - Win Rate @36 (%): 36手目時点での理論的な勝率
            - Come-From-Behind Win (%) : 36手目時点から逆転勝ちした割合
            - Come-From-Behind Lose (%): 36手目時点から逆転負けした割合
        """
    )
    df_rank_sorted = st.session_state['df_rank_sorted']
    st.dataframe(st.session_state['df_rank_sorted'])

    # 上位30位のプレイヤーのAverage Total LossとAverage 24 Empty Lossを折れ線グラフでプロット
    top30 = st.session_state['df_rank_sorted'].head(30)
    plt.figure(figsize=(10, 6))

    # Average Total Lossのプロット
    plt.plot(np.arange(1, 31, 1), top30['Average Total Loss'], marker='o', label='Average Total Loss')

    # Average 24 Empty Lossのプロット
    plt.plot(np.arange(1, 31, 1), top30['Average 24 Empty Loss'], marker='x', label='Endgame (37~)')
    plt.plot(np.arange(1, 31, 1), top30['Average Total Loss'] - top30['Average 24 Empty Loss'], marker='x', label='Midgame (~36)')

    plt.title('Top 30 Players Average Losses')
    plt.xlabel('Rank')
    plt.ylabel('Loss Value')
    plt.xticks(range(1, 31))  # 1位から30位までの目盛りを設定
    plt.ylim(0, max(top30['Average Total Loss'].max(), top30['Average 24 Empty Loss'].max()) + 1)  # y軸の最小値を0に設定
    plt.grid(True)
    plt.legend()  # 凡例の表示

    st.write(" ### Top 30人の平均石損")
    # Streamlitでプロットを表示
    st.pyplot(plt)

    st.write(" ### Top 10 Player の6手ごとの平均石損分布")
    # 可視化関数を呼び出し、図を取得
    plt = plot_loss_sixmoves_boxplot(st.session_state['df_loss'], st.session_state['top10_playerids'])

    # Streamlitで図を表示
    st.pyplot(plt)
    # グループの選択
    st.write("""
             ## プレイヤー間の比較
             二つのグループ間で石損指標の詳細を比較する。
             - ランキング範囲を指定: 上記ランキングに従ってグループを作成する
             - Player Idを直接選択: 特定のプレイヤーをIDで選択
             """)
    col1, col2 = st.columns(2)

    with col1:
        st.write("グループ1の選択")
        selection_method_group1 = st.radio(
            "選択方法を選んでください",
            ('ランキング範囲を指定', 'Player Idを直接選択'),
            key="selection_method_group1"
        )

        if selection_method_group1 == 'Player Idを直接選択':
            group1_player_ids = st.multiselect("Player Idを選択", options=st.session_state['df_rank_sorted']['Player Id'].to_list(), key="group1_player_ids")
        else:
            group1_rank_start = st.number_input("ランキング開始位置（n位）", min_value=1, max_value=len(st.session_state['df_rank_sorted']), key="group1_rank_start")
            group1_rank_end = st.number_input("ランキング終了位置（m位）", min_value=1, max_value=len(st.session_state['df_rank_sorted']), key="group1_rank_end")

    with col2:
        st.write("グループ2の選択")
        selection_method_group2 = st.radio(
            "選択方法を選んでください",
            ('ランキング範囲を指定', 'Player Idを直接選択'),
            key="selection_method_group2"
        )

        if selection_method_group2 == 'Player Idを直接選択':
            group2_player_ids = st.multiselect("Player Idを選択", options=st.session_state['df_rank_sorted']['Player Id'].to_list(), key="group2_player_ids")
        else:
            group2_rank_start = st.number_input("ランキング開始位置（n位）", min_value=1, max_value=len(st.session_state['df_rank_sorted']), key="group2_rank_start")
            group2_rank_end = st.number_input("ランキング終了位置（m位）", min_value=1, max_value=len(st.session_state['df_rank_sorted']), key="group2_rank_end")

    if st.button("グループ生成"):
        group1_ids, group2_ids = [], []
        if selection_method_group1 == 'Player Idを直接選択':
            group1_ids = group1_player_ids
        else:
            group1_ids = get_player_ids_from_ranking(st.session_state['df_rank_sorted'], group1_rank_start, group1_rank_end)

        if selection_method_group2 == 'Player Idを直接選択':
            group2_ids = group2_player_ids
        else:
            group2_ids = get_player_ids_from_ranking(st.session_state['df_rank_sorted'], group2_rank_start, group2_rank_end)

        col1, col2 = st.columns(2)
        with col1:
            # グループ1のPlayer Idに基づいてフィルタリング
            group1_df = df_rank_sorted.filter(pl.col('Player Id').is_in(group1_ids))
            st.write("グループ1のPlayer Id:")
            st.dataframe(group1_df)

        with col2:
            # グループ2のPlayer Idに基づいてフィルタリング
            group2_df = df_rank_sorted.filter(pl.col('Player Id').is_in(group2_ids))
            st.write("グループ2のPlayer Id:")
            st.dataframe(group2_df)

        st.write(" #### 6手ごとの平均loss分布")
        # 可視化関数を呼び出し、図を取得
        plt = plot_group_loss_sixmoves_boxplot(st.session_state['df_loss'], group1_ids, group2_ids)

        # Streamlitで図を表示
        st.pyplot(plt)

        # # Win Rateの散布図を表示
        # plt = scatter_group_winrate(df_rank_sorted[df_rank_sorted['Player Id'].isin(group1_ids)],df_rank_sorted[df_rank_sorted['Player Id'].isin(group2_ids)])

        # # Streamlitで図を表示
        # st.pyplot(plt)

        st.write(" #### 勝率分布")
        # group1_df = df_rank_sorted.filter(pl.col('Player Id').is_in(group1_ids))
        # group2_df = df_rank_sorted.filter(pl.col('Player Id').is_in(group2_ids))

        # インタラクティブな散布図を作成
        fig = scatter_group_winrate_interactive(group1_df, group2_df)
        # Streamlitで図を表示
        st.plotly_chart(fig)
