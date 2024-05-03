import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class OthelloCV:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_trim_cut, self.gray_image, self.resized_image = self.trim_image(self.image)
    
    def crop_green_rectangle(self, image):
        # HSV色空間に変換
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 緑色の範囲を定義 (これは調整が必要な場合があります)
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        
        # 緑色のマスクを作成
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        # マスクから輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 最大の輪郭を見つける
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 最大輪郭のバウンディングボックスを取得
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 元の画像から領域を切り抜く
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image
        else:
            # 緑の領域が見つからなかった場合
            return None
        
    def trim_image(self, image):
        image_trim = self.crop_green_rectangle(image)
        if image_trim is None:
            print("緑の領域が見つかりませんでした。")
            return None, None, None  # 適切な値を返すか、例外を投げる

        cut_ratio = 0.05
        trim_col = int(image_trim.shape[0] * cut_ratio)
        trim_row = int(image_trim.shape[1] * cut_ratio)
        image_trim_cut = image_trim[trim_col:image_trim.shape[0]-trim_col, trim_row:image_trim.shape[1]-trim_row, :]

        gray_image = cv2.cvtColor(image_trim_cut, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(gray_image, (8, 8))

        return image_trim_cut, gray_image, resized_image
    
    def convert_board_to_string(self):
        ret = ""
        for stone in self.resized_image.flatten():
            if stone > 215:
                ret += "O"
            elif stone < 40:
                ret += "X"
            else:
                ret += "-"
        return ret
    
    def convert_resized_to_hamlite(self, start_color="black"):
        ret = self.convert_board_to_string()

        hamlite_template = """
https://reversi-ai.appspot.com/v1.62/hamlite.html?&lastmove_check=on&lastmove_mark=num&start_board={start_board}&start_color={start_color}&ai=on
        """.format(start_board=ret, start_color=start_color)
        return hamlite_template
    
    def get_hamlite(self, start_color="black"):
        return self.convert_resized_to_hamlite(start_color=start_color)
    
    def get_image_trim_cut(self):
        return self.image_trim_cut
    
    def get_gray_image(self):
        return self.gray_image
    
    def get_resized_image(self):
        return self.resized_image
    
    def draw_othello_board_mono(self):
        board_string = self.convert_board_to_string()
        # 盤面のサイズとマスのサイズを設定
        cell_size = 50
        board_size = 8 * cell_size
        
        # 盤面画像を作成（グレースケールで中間色で初期化）
        board_img = np.full((board_size+1, board_size+1), 180, dtype=np.uint8)
        
        # 8x8の格子を描画
        for i in range(0, board_size, cell_size):
            for j in range(0, board_size, cell_size):
                cv2.rectangle(board_img, (i, j), (i + cell_size, j + cell_size), (0), 1)  # 黒で輪郭を描画
        
        # 文字列を解析して、黒丸または白丸を配置
        blacks = 0
        whites = 0
        for index, char in enumerate(board_string):
            row = index // 8
            col = index % 8
            center = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
            radius = int(cell_size / 2.5)  # 半径を少し大きく設定
            if char == 'X':
                cv2.circle(board_img, center, radius, (0), -1)  # 黒丸
                blacks += 1
            elif char == 'O':
                cv2.circle(board_img, center, radius, (255), -1)  # 白丸
                whites += 1

        # 描画用のfigオブジェクトを作成
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(board_img, cmap="binary_r", cbar=False, square=True, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_xlabel(f"⚫️{blacks}     ⚪️{whites}", size=30)  # ラベルを追加

        return fig