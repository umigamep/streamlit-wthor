import streamlit as st
import tempfile
import streamlit.components.v1 as components

from src.OthelloCV import OthelloCV

# 説明
st.write(
    """
    # オセロクエストのスクリーンショットをHamliteに変換
    スマホ版オセクエのスクショから、Hamliteのリンクを作成します。
    オセクエ以外のスクショでも動く可能性はありますが、基本はオセクエ向けです。
    """
)

# ユーザーからの画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=['png', 'jpg', 'jpeg'])

# 開始色の選択
start_color = st.radio("手番を選んでください:", options=["black", "white"], index=0)  # index 0 sets 'black' as the default

# ユーザが画像をアップロードした後の処理
if uploaded_file is not None:
    # 画像ファイルを一時保存
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    
    # 画像のpathを渡してOthelloCVを初期化
    othello_cv = OthelloCV(image_path=temp_file.name)
    temp_file.close()

    # 切り抜かれた画像を表示
    # fig = othello_cv.draw_othello_board_mono()
    # st.pyplot(fig)

    # HamliteのURLを作成
    url = othello_cv.get_hamlite(start_color=start_color)
    components.iframe(url, scrolling=True, width=300, height=400)
    #st.write(f"Hamlite URL: [{url}]({url})")
else:
    st.write("画像が選択されていません")
