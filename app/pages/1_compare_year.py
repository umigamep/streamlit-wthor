import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

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
    df_result = df_result.sort(['Average_Sum_Player_Loss', 'Record_Count'], descending=[True, False])

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
        pl.col('Record_Count').alias('# Games'),
        pl.col('Average_Sum_Player_Loss').alias('Average Total Loss'),
        pl.col('Average_Sum_Player_Loss_24Empty').alias('Average 24 Empty Loss'),
        pl.col('Win_Rate').alias('Win Rate (%)'),
        pl.col('Win_Rate_Theoretical').alias('Win Rate @36 (%)'),
        pl.col('Come-From-Behind Win (%)'),
        pl.col('Come-From-Behind Lose (%)')
    ])

    return df_rank_sorted



# TOP10プレイヤーの取得
def get_yearly_top_players(start_year, end_year, num_top=10):
    yearly_top10_players = {}
    for year in range(start_year, end_year + 1):
        # 1. データの読み込みと前処理
        df = load_and_process_data(year)  # Polars版のデータ読み込み関数
        df_id_name = make_df_id_name(df)  # Polars版のIDと名前のデータフレーム作成関数
        df_loss = make_df_loss(df, df_id_name)  # Polars版の損失データ集計関数

        # 2. 平均損失の計算
        df_avg_loss = calculate_average_sum_player_loss_with_name(df_loss, df_id_name, count_min=10)

        # 3. 特定の条件に基づいたフィルタリング
        df_avg_loss_filtered = df_avg_loss.filter(~pl.col('Player Name').str.contains("\("))

        # 4. 各年のトッププレイヤーの選択
        top10_players = df_avg_loss_filtered.sort('Average Total Loss').head(num_top)
        yearly_top10_players[year] = top10_players

    return yearly_top10_players

# 加重平均損失のプロット
def plot_weighted_average_loss_top_per_year(yearly_top_players, detail="なし"):
    years = list(yearly_top_players.keys())
    values = []
    values_24empty = []
    for year, top_players in yearly_top_players.items():
        weighted_avg = (top_players['Average Total Loss'] * top_players['# Games']).sum() / top_players['# Games'].sum()
        weighted_avg_24empty = (top_players['Average 24 Empty Loss'] * top_players['# Games']).sum() / top_players['# Games'].sum()
        values.append(weighted_avg)
        values_24empty.append(weighted_avg_24empty)

    values = np.array(values)
    values_24empty = np.array(values_24empty)

    plt.figure(figsize=(10, 5))
    if detail != "内訳のみ表示":
        plt.plot(years, values, marker='o', label="Average Total Loss", color="C0")
    if detail in ["あり", "内訳のみ表示"]:
        plt.plot(years, values_24empty, marker='o', label="Endgame (37~)", color="C1")
        plt.plot(years, values - values_24empty, marker='o', label="Midgame (~36)", color="C2")
    plt.title("Yearly Average Loss of Top Players")
    plt.xlabel("Year")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.xticks(years, rotation=45)
    plt.tight_layout()

    return plt


# Streamlitページの設定
st.title("トッププレイヤーの石損の変遷")

# ユーザーが選択できるスライダーの追加
num_top = st.slider('選択するトッププレイヤーの数', 1, 100, 10)
detail = st.radio(
    "石損の内訳を表示しますか",
    options=["なし", "あり", "内訳のみ表示"],
    index=0
)

yearly_top_players = get_yearly_top_players(2001, 2023, num_top=num_top)
plot = plot_weighted_average_loss_top_per_year(yearly_top_players, detail)
st.pyplot(plot)
# 各年のTOP10プレイヤーのデータを表示
for year, top in yearly_top_players.items():
    st.write(f"Top Players in {year}")
    st.dataframe(top)
