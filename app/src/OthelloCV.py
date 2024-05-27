import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class OthelloCV:
    def __init__(self, image_path):
        """
        Initialize the OthelloCV object with the given image path.
        """
        self.image = self.load_and_preprocess_image(image_path)
        self.image_trim_cut, self.gray_image, self.resized_image = self.trim_image(self.image)
    
    def load_and_preprocess_image(self, image_path):
        """
        Load the image from the given path and preprocess it.
        If the height is 1.5 times the width or more, crop the bottom 1/4 of the image.
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        if h >= 1.5 * w:
            image = image[:int(h * 0.75), :]
        return image
    
    def crop_green_rectangle(self, image):
        """
        Detect and crop the largest green rectangle from the image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 80, 80])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image
        return None
        
    def find_board_grid(self, image):
        """
        Detect the 8x8 board grid in the image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.lower_green = np.array([40, 80, 80])
        self.upper_green = np.array([90, 255, 200])
        mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        edges = cv2.Canny(mask, 50, 150, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            print("No lines detected.")
            return None
        line_image = np.zeros_like(image)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
        for line in lines:
            for x1, y1, x2, y2 in line:
                min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
                max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)
        cropped_image = image[min_y:max_y, min_x:max_x]
        return cropped_image
    
    def resize_with_mode(self, gray_image, new_size=(8, 8)):
        """
        Resize the grayscale image to the given size using the mode of each block.
        """
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

    def resize_by_2step(self, gray_image):
        """
        Resize the grayscale image in two steps to an 8x8 image.
        """
        resized_image_16 = cv2.resize(gray_image, (16, 16))
        resized_image = np.zeros((8, 8), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                block = resized_image_16[2*j:2*j+2, 2*i:2*i+2]
                if np.min(block) < 35:
                    resized_image[j, i] = 0
                elif np.max(block) > 220:
                    resized_image[j, i] = 255
                else:
                    resized_image[j, i] = 122
        return resized_image

    def trim_image(self, image):
        """
        Trim the image to the board grid and preprocess it.
        """
        image_trim = self.find_board_grid(image)
        if image_trim is None:
            print("緑の領域が見つかりませんでした。")
            return None, None, None
        cut_ratio = 0.02
        trim_col = int(image_trim.shape[0] * cut_ratio)
        trim_row = int(image_trim.shape[1] * cut_ratio)
        image_trim_cut = image_trim[trim_col:image_trim.shape[0]-trim_col, trim_row:image_trim.shape[1]-trim_row, :]
        gray_image = cv2.cvtColor(image_trim_cut, cv2.COLOR_RGB2GRAY)
        resized_image_cv = self.resize_by_2step(gray_image)
        resized_image_cv[resized_image_cv < 35] = 0
        resized_image_cv[resized_image_cv > 200] = 255
        resized_image = resized_image_cv
        return image_trim_cut, gray_image, resized_image
    
    def convert_board_to_string(self):
        """
        Convert the resized image to a string representing the board state.
        """
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
        """
        Convert the resized image to a HamLite URL.
        """
        ret = self.convert_board_to_string()
        hamlite_template = f"https://umigamep.github.io/othello_board_js/?start_board={ret}&start_color={start_color}"
        return hamlite_template
    
    def get_hamlite(self, start_color="black"):
        """
        Get the HamLite URL for the current board state.
        """
        return self.convert_resized_to_hamlite(start_color=start_color)
    
    def get_image_trim_cut(self):
        """
        Get the trimmed image.
        """
        return self.image_trim_cut
    
    def get_gray_image(self):
        """
        Get the grayscale image.
        """
        return self.gray_image
    
    def get_resized_image(self):
        """
        Get the resized image.
        """
        return self.resized_image
    
    def draw_othello_board_mono(self):
        """
        Draw the Othello board in monochrome.
        """
        board_string = self.convert_board_to_string()
        cell_size = 50
        board_size = 8 * cell_size
        board_img = np.full((board_size+1, board_size+1), 180, dtype=np.uint8)
        for i in range(0, board_size, cell_size):
            for j in range(0, board_size, cell_size):
                cv2.rectangle(board_img, (i, j), (i + cell_size, j + cell_size), (0), 1)
        blacks, whites = 0, 0
        for index, char in enumerate(board_string):
            row = index // 8
            col = index % 8
            center = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
            radius = int(cell_size / 2.5)
            if char == 'X':
                cv2.circle(board_img, center, radius, (0), -1)
                blacks += 1
            elif char == 'O':
                cv2.circle(board_img, center, radius, (255), -1)
                whites += 1
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(board_img, cmap="binary_r", cbar=False, square=True, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_xlabel(f"⚫️{blacks}     ⚪️{whites}", size=30)
        return fig
