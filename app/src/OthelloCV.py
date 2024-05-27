import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class OthelloCV:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_trim_cut, self.gray_image, self.resized_image = self.trim_image(self.image)
    
    def crop_green_rectangle(self, image):
        # HSV色空間に変換
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 緑色の範囲を定義 (これは調整が必要な場合があります)
        lower_green = np.array([40, 90, 90])
        upper_green = np.array([80, 255, 255])
        
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
        
    def find_board_grid(self, image):
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the range for green color (adjusted)
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([90, 255, 200])
        
        # Create a mask for the green color
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        # Apply edge detection to find the grid lines
        edges = cv2.Canny(mask, 50, 150, apertureSize=3, L2gradient=True)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            print("No lines detected.")
            return None
        
        # Create an empty image to draw lines on
        line_image = np.zeros_like(image)
        
        # Draw the detected lines on the empty image
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the image with detected lines
        # plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        # plt.title('Detected Lines')
        # plt.show()
        
        # Find the bounding box of the board based on detected lines
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
                max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)
        
        # Return the cropped image based on the bounding box
        cropped_image = image[min_y:max_y, min_x:max_x]
        
        # Display the cropped image
        # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # plt.title('Cropped Image')
        # plt.show()
    
        return cropped_image
    
    def resize_with_mode(self, gray_image, new_size=(8, 8)):
        h, w = gray_image.shape
        resized_image = np.zeros(new_size, dtype=np.uint8)
        block_size_x = w // new_size[0]
        block_size_y = h // new_size[1]

        for i in range(new_size[0]):
            for j in range(new_size[1]):
                block = gray_image[j*block_size_y:(j+1)*block_size_y, i*block_size_x:(i+1)*block_size_x]
                mode = stats.mode(block, axis=None)[0]
                resized_image[j, i] = mode
        return resized_image

    def trim_image(self, image):
        image_trim = self.find_board_grid(image) #self.crop_green_rectangle(image)
        if image_trim is None:
            print("緑の領域が見つかりませんでした。")
            return None, None, None  # 適切な値を返すか、例外を投げる

        cut_ratio = 0.03
        trim_col = int(image_trim.shape[0] * cut_ratio)
        trim_row = int(image_trim.shape[1] * cut_ratio)
        image_trim_cut = image_trim[trim_col:image_trim.shape[0]-trim_col, trim_row:image_trim.shape[1]-trim_row, :]

        gray_image = cv2.cvtColor(image_trim_cut, cv2.COLOR_RGB2GRAY)
        
        resized_image_cv = cv2.resize(gray_image, (8, 8))
        resized_image_cv[resized_image_cv < 50] = 0
        resized_image_cv[resized_image_cv > 200] = 255

        resized_image_mode = self.resize_with_mode(gray_image)
        resized_image_mode[resized_image_mode < 50] = 0
        resized_image_mode[resized_image_mode > 200] = 255

        # 新しい行列を作成
        resized_image = np.full((8, 8), 122, dtype=np.uint8)

        # どちらかが0ならば0
        resized_image[(resized_image_cv == 0) | (resized_image_mode == 0)] = 0

        # どちらかが255ならば255
        resized_image[(resized_image_cv == 255) | (resized_image_mode == 255)] = 255

        # plt.imshow(gray_image, cmap="gray")
        # plt.show()
        # print(resized_image)
        # plt.imshow(resized_image, cmap="gray")
        # plt.show()

        return image_trim_cut, gray_image, resized_image
    
    def convert_board_to_string(self):
        ret = ""
        for stone in self.resized_image.flatten():
            if stone > 200:
                ret += "O"
            elif stone < 60:
                ret += "X"
            else:
                ret += "-"
        return ret
    
    def convert_resized_to_hamlite(self, start_color="black"):
        ret = self.convert_board_to_string()

        hamlite_template = """
https://umigamep.github.io/othello_board_js/?start_board={start_board}&start_color={start_color}
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