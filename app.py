import cv2
import numpy as np
from PIL import Image

# 楕円検出
def detect_ellipses(image_path, min_size, max_size):
    # 画像を読み込む
    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 色の範囲を定義
    lower_color1 = np.array([0, 50, 50])
    upper_color1 = np.array([10, 255, 255])
    lower_color2 = np.array([170, 50, 50])
    upper_color2 = np.array([180, 255, 255])

# 二値化準備
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    mask = cv2.bitwise_or(mask1, mask2)

# 二値化
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # グレイスケール化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ガウシアンブラー
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # エッジ検出
    edges = cv2.Canny(result, 50, 150)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipses = []
    for contour in contours:
        if len(contour) >= 5:  # 楕円フィッティングinit
            ellipse = cv2.fitEllipse(contour)
            if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # 楕円の幅と高さが正の値
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    
    return ellipses, image

# レイヤー作成
def create_layer(image, ellipses):
    layer = np.zeros_like(image)
    
    for ellipse in ellipses:
        cv2.ellipse(layer, ellipse, (255, 255, 255), -1)
    
    return layer

# 変化を検出
def detect_changes(image1, image2, layer):
    # レイヤーを適用
    masked_image1 = cv2.bitwise_and(image1, layer)
    masked_image2 = cv2.bitwise_and(image2, layer)
    
    # 差分を計算
    diff = cv2.absdiff(masked_image1, masked_image2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    return diff_thresh

# メイン処理
def main():
    image1_path = 'base2.jpg'
    image2_path = 'test2.png'
    
    # 楕円検出のサイズ範囲を指定（ピクセル単位）
    min_size = 0  # 最小サイズ
    max_size = 1000  # 最大サイズ
    
    # 楕円検出
    ellipses, image1 = detect_ellipses(image1_path, min_size, max_size)
    
    # 検出された楕円を描画した画像を保存
    detected_image = image1.copy()
    for ellipse in ellipses:
        cv2.ellipse(detected_image, ellipse, (0, 255, 0), 2)  # 緑色で楕円を描画
    cv2.imwrite('detected_objects.png', detected_image)
    print(f"検出された楕円を 'detected_objects.png' として保存しました。")
    
    # レイヤー作成
    layer = create_layer(image1, ellipses)
    
    # 2つ目の画像を読み込む
    image2 = cv2.imread(image2_path)
    
    # 変化検出
    changes = detect_changes(image1, image2, layer)
    
    # 結果を保存
    cv2.imwrite('changes.png', changes)
    print(f"サイズ範囲 {min_size} から {max_size} の楕円を検出し、変化を 'changes.png' として保存しました。")

if __name__ == "__main__":
    main()