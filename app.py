import cv2
import numpy as np
from PIL import Image

# 楕円検出を行う関数
def detect_ellipses(image_path, min_size, max_size):
    # 画像を読み込む
    image = cv2.imread(image_path)
    
    # グレイスケール化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ガウシアンブラーを適用してノイズを除去
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # エッジ検出
    edges = cv2.Canny(blurred, 50, 150)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipses = []
    for contour in contours:
        if len(contour) >= 5:  # 楕円フィッティングには最低5点必要
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            
            # 楕円の長軸と短軸を取得
            major_axis = max(axes)
            minor_axis = min(axes)
            
            # サイズ範囲内の楕円のみを抽出
            if min_size <= major_axis <= max_size and min_size <= minor_axis <= max_size:
                ellipses.append(ellipse)
    
    return ellipses, image

# レイヤーを作成する関数
def create_layer(image, ellipses):
    layer = np.zeros_like(image)
    
    for ellipse in ellipses:
        cv2.ellipse(layer, ellipse, (255, 255, 255), -1)
    
    return layer

# 変化を検出する関数
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
    image1_path = 'base1.jpg'
    image2_path = 'test2.png'
    
    # 楕円検出のサイズ範囲を指定（ピクセル単位）
    min_size = 17  # 最小サイズ
    max_size = 50  # 最大サイズ
    
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