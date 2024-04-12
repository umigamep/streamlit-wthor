import streamlit as st

# Page Config
st.set_page_config(
    page_title="Wthor Visualization",
    page_icon=":croun:"
)

# Title
st.title("WTHOR データベース可視化ツール")

st.write(
    """
    (開発中)

    WTHOR データベースの棋譜を集計・AIで分析した結果を可視化するWebアプリです。

    左のタブから分析観点を選べます。
    
<<<<<<< HEAD
    - player_stats: プレイヤーの詳細な統計値を調べられます
    - ranking: 統計値のランキングを調べられます
=======
    - year analysis: 年ごとのトッププレイヤーの石損の指標を比較します
        - 改修が遅れていて、終盤の指標が24個空きのものになっています
    - player analysis: プレイヤーごとの石損指標を比較します
    - search games: (開発中)着手の統計情報や棋譜からゲームを検索する機能です
>>>>>>> 14fc402efc70c61534fdd9a753b3e576a9dcd133

    ### 参考リンク

    - [WTHOR データベース](https://www.ffothello.org/informatique/la-base-wthor/)
        - フランスオセロ連盟が公開している棋譜データベース
    - [オセロの棋譜データベース WTHOR の読み込み方](https://qiita.com/tanaka-a/items/e21d32d2931a24cfdc97)
        - wthor 形式のファイルを CSV に変換するのに活用させていただきました
    - [edax](https://github.com/abulmo/edax-reversi)
        - 強力なオセロAI。解析に Lv18 + 手持ちの book を使用しました
    
    ### 注意
    
    - WTHOR データベースに含まれる対局には一定のバイアスがあることに留意してご活用ください
        - 現在は2001年以降のデータが用いられています
        - 2023年のデータは2023/12/8に取得したものを用いています
    - AIの評価は必ずしも正確で無い場合があります
    - データベース内で同一人物(id)に複数の名前が登録されている場合に、結果の表示に現在の名前が反映されないことがあります

    """
)