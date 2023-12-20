import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def convert_string_to_list(string):
    return [int(x.strip()) for x in string.strip('[]\n').split(' ') if x.strip().isdigit()]

# CSVファイルの読み込みと前処理
def load_and_process_data(year):
    file_path = f"app/resources/addloss_wthor_{year}.csv"
    df_year = pd.read_csv(file_path)
    df_year['loss_black'] = df_year['loss_black'].apply(convert_string_to_list)
    df_year['loss_white'] = df_year['loss_white'].apply(convert_string_to_list)
    df_year['year'] = year
    return df_year

# IDと名前のDataFrameを作成
def make_df_id_name(df):
    df_black = df[['blackPlayerId', 'blackPlayerName']].rename(columns={'blackPlayerId': 'PlayerId', 'blackPlayerName': 'PlayerName'}).drop_duplicates()
    df_white = df[['whitePlayerId', 'whitePlayerName']].rename(columns={'whitePlayerId': 'PlayerId', 'whitePlayerName': 'PlayerName'}).drop_duplicates()
    return pd.concat([df_black, df_white]).drop_duplicates()

# 損失データの作成
def make_df_loss(df, df_id_name):
    rows = []
    for id in df_id_name['PlayerId']:
        for _, row in df[df['blackPlayerId'] == id].iterrows():
            rows.append(create_loss_row(id, row, 'black'))
        for _, row in df[df['whitePlayerId'] == id].iterrows():
            rows.append(create_loss_row(id, row, 'white'))

    df_loss = pd.DataFrame(rows)
    df_loss["Sum_Player_Loss"] = df_loss["Player_Loss"].apply(sum)
    df_loss["Sum_Player_Loss_24Empty"] = df_loss["Player_Loss"].apply(lambda x: sum(x[-24:]))
    return df_loss

# Loss Rowの作成
def create_loss_row(id, row, color):
    if color == 'black':
        score = row['blackScore']
        theoretical_score = row['blackTheoreticalScore']
        loss = row['loss_black']
        opponent_loss = row['loss_white']
    else:
        score = 64 - row['blackScore']  # whiteScoreを計算
        theoretical_score = 64 - row['blackTheoreticalScore']  # whiteTheoreticalScoreを計算
        loss = row['loss_white']
        opponent_loss = row['loss_black']

    return {
        'PlayerId': id,
        'OpponentId': row['whitePlayerId'] if color == 'black' else row['blackPlayerId'],
        'TournamentId': row['tournamentId'],
        'Color': 0 if color == 'black' else 1,
        'Score': score,
        'TheoreticalScore': theoretical_score,
        'Player_Loss': loss,
        'Opponent_Loss': opponent_loss,
        'Transcript': row['transcript'],
        'Year': row['year']
    }

def calculate_result(score):
    if score > 32:
        return 1
    elif score == 32:
        return 0.5
    else:
        return 0
# 平均損失の計算
def calculate_average_sum_player_loss_with_name(df_loss, df_id_name, count_min=10):
    # ScoreとTheoreticalScoreの結果を計算
    df_loss['Win'] = df_loss['Score'].apply(calculate_result)
    df_loss['Win Theoretical'] = df_loss['TheoreticalScore'].apply(calculate_result)

    # PlayerId ごとにグループ化し、Sum_Player_LossとWin Rateの平均値とカウントを計算
    grouped = df_loss.groupby('PlayerId').agg({'Sum_Player_Loss': ['mean', 'count'],'Sum_Player_Loss_24Empty': 'mean' ,'Win': 'mean', 'Win Theoretical': 'mean'})

    # カラム名を変更
    grouped.columns = ['Average_Sum_Player_Loss', 'Record_Count', 'Average_Sum_Player_Loss_24Empty', 'Win_Rate', 'Win_Rate_Theoretical']

    # 'Record_Count' が count_min より大きいレコードのみを対象にする
    filtered_grouped = grouped[grouped['Record_Count'] > count_min]

    # DataFrame をリセットして PlayerId をカラムに戻す
    df_result = filtered_grouped.reset_index()

    # df_id_name で PlayerId ごとに最初の PlayerName を取得
    df_id_name_unique = df_id_name.drop_duplicates(subset='PlayerId')

    # df_result と df_id_name_unique をマージ
    df_result = df_result.merge(df_id_name_unique, on='PlayerId', how='left')

    # 'Average_Sum_Player_Loss' の小さい順に、同じ値の場合は 'Record_Count' の大きい順に並び替え
    df_result = df_result.sort_values(by=['Average_Sum_Player_Loss', 'Record_Count'], ascending=[True, False])

    # Average_Sum_Player_Loss を小数点以下第2位までに丸める
    df_result['Average_Sum_Player_Loss'] = df_result['Average_Sum_Player_Loss'].round(2)
    df_result['Average_Sum_Player_Loss_24Empty'] = df_result['Average_Sum_Player_Loss_24Empty'].round(2)

    # Win Rateをパーセント表記で小数点以下1桁に丸める
    df_result['Win_Rate'] = (df_result['Win_Rate'] * 100).round(1)
    df_result['Win_Rate_Theoretical'] = (df_result['Win_Rate_Theoretical'] * 100).round(1)

    # 'Come From Behind' の計算
    # WinからWin Theoreticalを引いた値が正のものについてのみの合計を計算
    df_loss['Come From Behind W'] = (df_loss['Win'] - df_loss['Win Theoretical']).clip(lower=0)
    df_loss['Come From Behind L'] = (df_loss['Win Theoretical'] - df_loss['Win']).clip(lower=0)
    # 各プレイヤーごとに 'Come From Behind W' と 'Come From Behind L' の合計を計算
    come_from_behind_grouped = df_loss.groupby('PlayerId').agg({
        'Come From Behind W': 'sum',
        'Come From Behind L': 'sum'
    })
    # df_result とマージし、試合数で割ってパーセント表示に変換
    df_result = df_result.merge(come_from_behind_grouped, on='PlayerId', how='left')
    df_result['Come-From-Behind Win (%)'] = (df_result['Come From Behind W'] / df_result['Record_Count'] * 100).round(1)
    df_result['Come-From-Behind Lose (%)'] = (df_result['Come From Behind L'] / df_result['Record_Count'] * 100).round(1)

    # df_rank_sorted のカラム順を変更し、indexを降順に振り直す
    df_rank_sorted = df_result[['PlayerId', 'PlayerName', 'Record_Count', 'Average_Sum_Player_Loss', 'Average_Sum_Player_Loss_24Empty', 'Win_Rate', 'Win_Rate_Theoretical', 'Come-From-Behind Win (%)', 'Come-From-Behind Lose (%)']]

    # カラム名を変更
    df_rank_sorted = df_rank_sorted.rename(columns={
        'PlayerId': 'Player Id',
        'PlayerName': 'Player Name',
        'Record_Count': '# Games',
        'Average_Sum_Player_Loss': 'Average Total Loss',
        'Average_Sum_Player_Loss_24Empty': 'Average 24 Empty Loss',
        'Win_Rate': 'Win Rate (%)',
        'Win_Rate_Theoretical': 'Win Rate @36 (%)'
    }).reset_index(drop=True)

    return df_rank_sorted
# TOP10プレイヤーの取得
def get_yearly_top10_players(start_year, end_year):
    yearly_top10_players = {}
    for year in range(start_year, end_year+1):
        df = load_and_process_data(year)
        df_id_name = make_df_id_name(df)
        df_loss = make_df_loss(df, df_id_name)
        df_avg_loss = calculate_average_sum_player_loss_with_name(df_loss, df_id_name, count_min=10)
        # Player Nameに"("が含まれるプレイヤーを除外
        df_avg_loss_filtered = df_avg_loss[~df_avg_loss['Player Name'].str.contains("\(")]

        # Top10のプレイヤーを選択し、インデックスをリセット
        top10_players = df_avg_loss_filtered.nsmallest(10, 'Average Total Loss').reset_index(drop=True)

        yearly_top10_players[year] = top10_players

    return yearly_top10_players

# 加重平均損失のプロット
def plot_weighted_average_loss_top10_per_year(yearly_top10_players):
    years = list(yearly_top10_players.keys())
    values = []
    values_24empty = []
    for year, top10_players in yearly_top10_players.items():
        weighted_avg = (top10_players['Average Total Loss'] * top10_players['# Games']).sum() / top10_players['# Games'].sum()
        weighted_avg_24empty = (top10_players['Average 24 Empty Loss'] * top10_players['# Games']).sum() / top10_players['# Games'].sum()
        values.append(weighted_avg)
        values_24empty.append(weighted_avg_24empty)

    values = np.array(values)
    values_24empty = np.array(values_24empty)

    plt.figure(figsize=(10, 5))
    plt.plot(years, values, marker='o', label = "Average Total Loss")
    # plt.plot(years, values_24empty, marker='o', label = "Endgame (37~)")
    # plt.plot(years, values - values_24empty, marker='o', label = "Midgame (~36)")
    plt.title("Yearly Average Loss of Top 10 Players")
    plt.xlabel("Year")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.xticks(years, rotation=45)
    plt.tight_layout()

    return plt

# Streamlitページの設定
st.title("Top 10 プレイヤーの石損の変遷")

yearly_top10_players = get_yearly_top10_players(2001, 2023)
plot = plot_weighted_average_loss_top10_per_year(yearly_top10_players)
st.pyplot(plot)
# 各年のTOP10プレイヤーのデータを表示
for year, top10 in yearly_top10_players.items():
    st.write(f"Top 10 Players in {year}")
    st.dataframe(top10)