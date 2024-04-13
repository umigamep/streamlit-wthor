import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

class WthorHandler:
    
    def __init__(self, years):
        all_year_df = None
        for year in years:
            df_year = self.load_and_process_data(year)
            if all_year_df is None:
                all_year_df = df_year
            else:
                all_year_df = pl.concat([all_year_df, df_year])
        self.all_year_df = all_year_df
        self.create_id_name_df(all_year_df)
        self.create_loss_df(all_year_df)

    def is_int(self, a):
        try:
            int(a)
        except:
            return False
        return True
    
    def convert_iswhite_to_list(self,string):
        return np.array([int(x.strip()) for x in string.strip('[]\n').split(', ') if self.is_int(x.strip())])
    
    def convert_blackscores_to_list(self,string):
        lst = np.array([int(x.strip()) for x in string.strip('[]\n').split(', ') if self.is_int(x.strip())])
        if len(lst) < 60:
            return np.array(lst.tolist() + [lst[-1]] * (60 - len(lst)))
        else:
            return lst
    
    def convert_loss_color_to_list(self,string):
        lst = np.array([int(x.strip()) for x in string.strip('[]\n').split(' ') if self.is_int(x.strip())])
        if len(lst) < 60:
            return np.array(lst.tolist() + [0] * (60 - len(lst)))
        else:
            return lst
    
    def load_and_process_data(self,year):
        csv_path = f"../app/resources/addloss_wthor_{year}.csv"
        df = pl.read_csv(csv_path)
        df = df.with_columns([
            pl.col('loss_black').map_elements(self.convert_loss_color_to_list).alias('loss_black'),
            pl.col('loss_white').map_elements(self.convert_loss_color_to_list).alias('loss_white'),
            pl.col('blackscores').map_elements(self.convert_blackscores_to_list).alias('blackscores'),
            pl.col('is_white_turn').map_elements(self.convert_iswhite_to_list).alias('is_white_turn'),
            pl.lit(year).alias('year')
        ])
        return df
    
    def cast_score_to_iswin(self, score):
        if score > 0:
            return 1
        elif score == 0:
            return 0
        return -1
    
    def cast_scores_to_iswin(self, scores):
        return np.array([self.cast_score_to_iswin(score) for score in scores])
    
    def convert_iswin_to_wld(self, score_list):
        return -np.diff(score_list)/2
    
    def create_id_name_df(self, df):
        df_black = df.select([
            pl.col('blackPlayerId').alias('PlayerId'),
            pl.col('blackPlayerName').alias('PlayerName')
        ]).unique()
        df_white = df.select([
            pl.col('whitePlayerId').alias('PlayerId'),
            pl.col('whitePlayerName').alias('PlayerName')
        ]).unique()
        self.id_name_df = pl.concat([df_black, df_white]).unique().sort(pl.col('PlayerId'))

    def create_loss_df(self, df):
        # isBlack = 1のdfを作成
        df_black = df.select([
            pl.col('blackPlayerId').alias('PlayerId'),
            pl.col('whitePlayerId').alias('OpponentId'),
            pl.lit(1).alias('IsBlack'),
            pl.col('loss_black').alias('PlayerLoss'), # 1手目から60手目
            pl.col('loss_white').alias('OpponentLoss'), # 1手目から60手目
            pl.col('blackscores').alias('PlayerScore'), # 0手目から59手目
            pl.col('blackscores').map_elements(self.cast_scores_to_iswin).alias('PlayerIsWin'), # 0手目から59手目
            pl.col('blackscores').map_elements(self.cast_scores_to_iswin).map_elements(self.convert_iswin_to_wld).alias('PlayerWLD'), # 1手目から59手目
            pl.col('transcript').alias('Transcript'),
            pl.col('tournamentId').alias('TournamentId'),
            pl.col('year').alias('Year')
        ])

        df_white = df.select([
            pl.col('whitePlayerId').alias('PlayerId'),
            pl.col('blackPlayerId').alias('OpponentId'),
            pl.lit(0).alias('IsBlack'),
            pl.col('loss_white').alias('PlayerLoss'), # 1手目から60手目
            pl.col('loss_black').alias('OpponentLoss'), # 1手目から60手目
            pl.col('blackscores').map_elements(lambda x: -x).alias('PlayerScore'), # 0手目から59手目
            pl.col('blackscores').map_elements(lambda x: -x).map_elements(self.cast_scores_to_iswin).alias('PlayerIsWin'), # 0手目から59手目
            pl.col('blackscores').map_elements(lambda x: -x).map_elements(self.cast_scores_to_iswin).map_elements(self.convert_iswin_to_wld).alias('PlayerWLD'), # 1手目から59手目
            pl.col('transcript').alias('Transcript'),
            pl.col('tournamentId').alias('TournamentId'),
            pl.col('year').alias('Year')
        ])
        self.loss_df = pl.concat([df_black, df_white]).sort(pl.col('PlayerId'))

    def get_all_year_df(self):
        return self.all_year_df
    
    def get_id_name_df(self):
        return self.id_name_df
    
    def get_loss_df(self):
        return self.loss_df
    
    def write_loss_df(self, save_path):
        func = lambda x: f"{x}"
        df = self.loss_df.with_columns([
            pl.col('PlayerLoss').map_elements(func).alias('PlayerLoss'),
            pl.col('OpponentLoss').map_elements(func).alias('OpponentLoss'),
            pl.col('PlayerScore').map_elements(func).alias('PlayerScore'),
            pl.col('PlayerIsWin').map_elements(func).alias('PlayerIsWin'),
            pl.col('PlayerWLD').map_elements(func).alias('PlayerWLD'),
        ])
        df.write_csv(save_path)
    
class LossCSVHandler:
    def __init__(self, path):
        if len(path)>0:
            df = pl.read_csv(path)
            self.loss_df = df.with_columns([
                pl.col('PlayerLoss').map_elements(self.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerLoss'),
                pl.col('OpponentLoss').map_elements(self.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('OpponentLoss'),
                pl.col('PlayerScore').map_elements(self.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerScore'),
                pl.col('PlayerIsWin').map_elements(self.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerIsWin'),
                pl.col('PlayerWLD').map_elements(self.convert_lossdf_array_to_list, return_dtype=pl.List(pl.Float64)).alias('PlayerWLD'),
            ])

        self.n_moves = 10  # 10手ごとにまとめる
        self.n_blocks = 6  # 6つのレンジでまとめる

        self.quantile_columns = [
            'PlayerTotalLossOver10Moves',
            'OpponentTotalLossOver10Moves',
        ]
        self.array_columns = self.quantile_columns + [
            'PlayerTotalWLDOver10Moves',
            'OpponentTotalWLDOver10Moves',
            'PlayerSigmoidLossOver10Moves',
        ]

        self.performance_df = None
        self.expand_df = None
        self.player_stas_df = None
    
    def is_float(self, element: any) -> bool:
        #If you expect None to be passed:
        if element is None: 
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False
    
    def convert_lossdf_array_to_list(self,string):
        return [float(x.strip()) for x in string.strip('[]\n').split(' ') if self.is_float(x.strip())]
    
    def loss_sum_over_n_moves(self, loss_array, n_moves=10):
        n_blocks = int(60/n_moves)
        average_loss_array = pl.Series([(loss_array[i*n_moves: (i+1)*n_moves]).sum() for i in range(n_blocks)])
        return average_loss_array
    
    def calc_players_performance(self, players, years):
        df = self.loss_df.filter(pl.col('PlayerId').is_in(players) & pl.col('Year').is_in(years))
        df = df.with_columns([
            pl.col('PlayerLoss').map_elements(lambda x: self.loss_sum_over_n_moves(x, n_moves=self.n_moves)).alias('PlayerTotalLossOver10Moves'),
            pl.col('OpponentLoss').map_elements(lambda x: self.loss_sum_over_n_moves(x, n_moves=self.n_moves)).alias('OpponentTotalLossOver10Moves'),
            pl.col('PlayerWLD').map_elements(lambda x: (np.abs(x)+np.array(x))/2.0).map_elements(lambda x: self.loss_sum_over_n_moves(x, n_moves=self.n_moves)).alias('PlayerTotalWLDOver10Moves'),
            pl.col('PlayerWLD').map_elements(lambda x: (np.abs(x)-np.array(x))/2.0).map_elements(lambda x: self.loss_sum_over_n_moves(x, n_moves=self.n_moves)).alias('OpponentTotalWLDOver10Moves'),
            pl.struct(['PlayerIsWin', 'IsBlack']).map_elements(lambda x: x['PlayerIsWin'][39] + 0.01*x['IsBlack'] > 0, return_dtype=pl.Int64).alias('PlayerIsWin@40'),
            pl.struct(['PlayerIsWin', 'IsBlack']).map_elements(lambda x: x['PlayerIsWin'][59] + 0.01*x['IsBlack'] > 0, return_dtype=pl.Int64).alias('PlayerIsWin@60'),
            pl.struct(['PlayerIsWin', 'IsBlack']).map_elements(
                lambda x: (x['PlayerIsWin'][59] + 0.01*x['IsBlack'] > 0) & (x['PlayerIsWin'][39] + 0.01*x['IsBlack'] > 0), return_dtype=pl.Int64).alias('W@60W@40'),  # 黒引き分け勝ち
            pl.struct(['PlayerIsWin', 'IsBlack']).map_elements(
                lambda x: (x['PlayerIsWin'][59] + 0.01*x['IsBlack'] > 0) & (x['PlayerIsWin'][39] + 0.01*x['IsBlack'] <= 0), return_dtype=pl.Int64).alias('W@60L@40'),
            pl.struct(['PlayerIsWin', 'IsBlack']).map_elements(
                lambda x: (x['PlayerIsWin'][59] + 0.01*x['IsBlack'] <= 0) & (x['PlayerIsWin'][39] + 0.01*x['IsBlack'] > 0), return_dtype=pl.Int64).alias('L@60W@40'),
            pl.struct(['PlayerIsWin', 'IsBlack']).map_elements(
                lambda x: (x['PlayerIsWin'][59] + 0.01*x['IsBlack'] <= 0) & (x['PlayerIsWin'][39] + 0.01*x['IsBlack'] <= 0), return_dtype=pl.Int64).alias('L@60L@40'),
            pl.col('PlayerLoss').map_elements(lambda x: x[39:].sum() == 0, return_dtype=pl.Int64).alias('PerfectEndGame'),
            pl.col('PlayerScore').map_elements(
                lambda x: np.concatenate(([0], np.diff(1 / (1 + np.exp(-x / 6)))))
            ).map_elements(
                lambda x: [(np.abs(v)-v)/2 for v in x]
            ).map_elements(lambda x: self.loss_sum_over_n_moves(x, n_moves=self.n_moves)).alias('PlayerSigmoidLossOver10Moves')
        ])
    
        self.performance_df = df
        return df
    
    def expand_array_column(self, df, column, size):
        # 一度に追加するとなぜかうまくいかないので、一旦逐次追加する
        # return df.with_columns([pl.col(column).map_elements(lambda x: x[i]).alias(f"{column}_{i}") for i in range(size)])
        for i in range(size):
            df = df.with_columns([pl.col(column).map_elements(lambda x: x[i]).alias(f"{column}_{i}")])
        return df

    def create_expand_df(self, players, years):
        df = self.calc_players_performance(players, years)
        # 配列をカラムに展開
        
        for col in self.array_columns:
            df = self.expand_array_column(df, column=col, size=self.n_blocks)

        self.expand_df = df
        return df
    
    def calc_players_stats(self, players, years):
        df = self.create_expand_df(players, years)

        stats_columns = [
            'PlayerIsWin@60',
            'PlayerIsWin@40',
            'W@60W@40',
            'W@60L@40',
            'L@60W@40',
            'L@60L@40',
            'PerfectEndGame'
            ]
        
        df = df.group_by(["PlayerId", "IsBlack"]).agg([
            pl.col('PlayerId').count().alias('count')
            ] + [
                pl.col(col).mean().alias(f"{col}_ratio") for col in stats_columns
            ] + [
                pl.col(f"{col}_{i}").mean().alias(f"{col}_{i}_mean") for col in self.array_columns for i in range(self.n_blocks)
            ] + [
                pl.col(f"{col}_{i}").quantile(0.5).alias(f"{col}_{i}_quant50") for col in self.quantile_columns for i in range(self.n_blocks)
            ] + [
                pl.col(f"{col}_{i}").quantile(0.75).alias(f"{col}_{i}_quant75") for col in self.quantile_columns for i in range(self.n_blocks)
            ] + [
                pl.col(f"{col}_{i}").quantile(0.25).alias(f"{col}_{i}_quant25") for col in self.quantile_columns for i in range(self.n_blocks)
            ]).sort(["PlayerId", "IsBlack"],descending=[False, True])
        
        self.player_stas_df = df
        return df
    
    def create_basic_stats_df(self, players, years, player_id_flg=0):
        df_players_stats = self.calc_players_stats(players=players, years=years)
        selected_col = [
                pl.col('IsBlack').map_elements(lambda x: ['White', 'Black'][x]).alias('Color'),
                pl.col('count').alias('# Games'),
                pl.col('PlayerIsWin@60_ratio').map_elements(lambda x: round(x*100,1)).alias('WinRate'),
                pl.col('PlayerIsWin@40_ratio').map_elements(lambda x: round(x*100,1)).alias('WinRate@40'),
                pl.col('W@60W@40_ratio').map_elements(lambda x: round(x*100,1)).alias("W@60,W@40"),
                pl.col('W@60L@40_ratio').map_elements(lambda x: round(x*100,1)).alias("W@60,L@40"),
                pl.col('L@60W@40_ratio').map_elements(lambda x: round(x*100,1)).alias("L@60,W@40"),
                pl.col('L@60L@40_ratio').map_elements(lambda x: round(x*100,1)).alias("L@60,L@40"),
                pl.col('PerfectEndGame_ratio').map_elements(lambda x: round(x*100,1)).alias('PerfectEndgame'),
                (-pl.col('PlayerTotalLossOver10Moves_4_mean')-pl.col('PlayerTotalLossOver10Moves_5_mean')).map_elements(lambda x: round(x,2)).alias('EndgameLoss'),
                (-pl.col('PlayerTotalWLDOver10Moves_4_mean')-pl.col('PlayerTotalWLDOver10Moves_5_mean')).map_elements(lambda x: round(x,2)).alias('EndgameWLDLoss'),
                (-pl.col('PlayerSigmoidLossOver10Moves_4_mean')-pl.col('PlayerSigmoidLossOver10Moves_5_mean')).map_elements(lambda x: round(x,3)).alias('EndgameSigmoidLoss'),
            ]
        if player_id_flg:
            selected_col = [pl.col('PlayerId').alias('PlayerId')] + selected_col

        df_basic = df_players_stats.select(selected_col)
        return df_basic

    def create_game_history_df(self, player, years):
        df_id_name = pl.read_csv("app/resources/id_name.csv")
        df_game_history = self.expand_df.filter(pl.col('Year').is_in(years)).join(
        df_id_name.with_columns([
            pl.col("PlayerId").alias('OpponentId'),
            pl.col("PlayerName").alias('OpponentName')
        ]), 
        on="OpponentId", how="left").select([
            pl.struct(['IsBlack','OpponentName']).map_elements(lambda x: player if x['IsBlack']==1 else x['OpponentName']).alias('BlackPlayer'),
            pl.struct(['IsBlack','OpponentName']).map_elements(lambda x: player if x['IsBlack']==0 else x['OpponentName']).alias('WhitePlayer'),
            pl.struct(['IsBlack','PlayerScore']).map_elements(lambda x: f"{32+x['PlayerScore'][-1]/2:.0f}-{32-x['PlayerScore'][-1]/2:.0f}" if x['IsBlack']==1 else f"{32-x['PlayerScore'][-1]/2:.0f}-{32+x['PlayerScore'][-1]/2:.0f}").alias('Result'),
            (-pl.col('PlayerTotalLossOver10Moves_4')-pl.col('PlayerTotalLossOver10Moves_5')).map_elements(lambda x: round(x,2)).alias('EndgameLoss'),
            (-pl.col('PlayerTotalWLDOver10Moves_4')-pl.col('PlayerTotalWLDOver10Moves_5')).map_elements(lambda x: round(x,2)).alias('EndgameWLDLoss'),
            (-pl.col('PlayerSigmoidLossOver10Moves_4')-pl.col('PlayerSigmoidLossOver10Moves_5')).map_elements(lambda x: round(x,3)).alias('EndgameSigmoidLoss'),
            pl.col('Year'),
            pl.col('TournamentId'),
            pl.col('Transcript'),
        ]).sort(['Year', 'TournamentId'])

        self.df_game_history = df_game_history
        return df_game_history
    # plot系
    def plot_PlayerLossOver10Moves(self, player, years, loss_column):
        df = self.create_expand_df([player], years)
        
        np_black = df.filter(pl.col('IsBlack') == 1).select([f"{loss_column}_{i}" for i in range(self.n_blocks)]).to_numpy()
        np_white = df.filter(pl.col('IsBlack') == 0).select([f"{loss_column}_{i}" for i in range(self.n_blocks)]).to_numpy()

        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(1,2,1)
        sns.boxplot(np_black, color='gray', ax=ax1, fill=None, showfliers=False)
        ax1.plot(np_black.mean(axis=0), 'd', label='average', color="red", markersize=6)
        ax1.set_xticklabels(['1~10','11~20','21~30','31~40','41~50','51~60']) 
        ax1.set_ylabel('Total Loss')
        ax1.set_title(f"Black ({len(np_black)} games)")
        ax1.grid(ls=":")
        ax1.legend()

        ax2 = fig.add_subplot(1,2,2)
        sns.boxplot(np_white, color='gray', ax=ax2, fill=None, showfliers=False)
        ax2.plot(np_white.mean(axis=0), 'd', label='average', color="red", markersize=6)
        ax2.set_xticklabels(['1~10','11~20','21~30','31~40','41~50','51~60']) 
        ax2.set_title(f"White ({len(np_white)} games)")
        ax2.grid(ls=":")
        ax2.legend()

        return fig
    # set系
    def set_loss_df(self, loss_df):
        self.loss_df = loss_df
    # get系
    def get_loss_df(self):
        return self.loss_df

    def get_performance_df(self):
        return self.performance_df
    
    def get_expand_df(self):
        return self.expand_df
    
    def get_player_stats_df(self):
        return self.player_stas_df
    