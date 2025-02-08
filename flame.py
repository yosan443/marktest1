import cv2
import numpy as np

# 画像を読み込む
image = cv2.imread('base2.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 色の範囲を定義（赤色の範囲を広く設定）
lower_color1 = np.array([0, 50, 50])
upper_color1 = np.array([10, 255, 255])
lower_color2 = np.array([170, 50, 50])
upper_color2 = np.array([180, 255, 255])

# 色範囲に含まれるピクセルを白、その他を黒にする
mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
mask = cv2.bitwise_or(mask1, mask2)

# マスクを適用して元の画像を二値化
result = cv2.bitwise_and(image, image, mask=mask)

# 二値化した画像を表示
cv2.imshow('Threshold Image', mask)

# 輪郭を検出
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 楕円をフィッティング
for contour in contours:
    if len(contour) >= 5:  # 楕円フィッティングには少なくとも5つの点が必要
        ellipse = cv2.fitEllipse(contour)
        if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # 楕円の幅と高さが正の値であることを確認
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)

# 結果を表示
cv2.imshow('Detected Ellipses', image)
cv2.waitKey(0)
cv2.destroyAllWindows()