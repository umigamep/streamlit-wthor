import streamlit as st

# Page Config
st.set_page_config(
    page_title="Wthor Visualization",
    page_icon=":croun:"
)

# Title
st.title("(開発中)wthorデータベース可視化ツール")

st.write(
    """
    wthorデータベースの棋譜を集計・分析した結果を可視化するWebアプリです。

    左のタブから分析観点を選べます。
    
    - compare_year: 年ごとのトッププレイヤーの石損の指標を比較します
    - compare_palyer: プレイヤーごとの石損指標を比較します
    - search_games: (開発中)着手の統計情報や棋譜からゲームを検索する機能です

    ### 参考リンク

    - [wthorデータベース](https://www.ffothello.org/informatique/la-base-wthor/)
        - フランスオセロ連盟が公開している棋譜データベース
    - [edax](https://github.com/abulmo/edax-reversi)
        - 解析にLv18+手持ちのbookを使用しました
    

    """
)