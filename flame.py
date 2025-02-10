import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import KDTree
import math

image = cv2.imread('base2.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 色の範囲を定義
lower_color1 = np.array([0, 50, 50])
upper_color1 = np.array([10, 255, 255])
lower_color2 = np.array([170, 50, 50])
upper_color2 = np.array([180, 255, 255])

# ２値化の前準備
mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
mask = cv2.bitwise_or(mask1, mask2)

# マスクから二値化
result = cv2.bitwise_and(image, image, mask=mask)

# 画像を表示
# cv2.imshow('Threshold Image', mask)

# 輪郭検出
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 楕円の中心座標を取得
centers = []
for contour in contours:
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        center = ellipse[0]
        if math.isfinite(center[0]) and math.isfinite(center[1]):
            centers.append(center)

# KDTreeを作成
tree = KDTree(centers)

def process_contour(contour):
    if len(contour) >= 5:  # 楕円フィッティングinit
        ellipse = cv2.fitEllipse(contour)
        if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # 楕円が正の範囲である
            min_size = (10, 25)  # 最小サイズ (幅, 高さ)
            max_size = (20, 30)  # 最大サイズ (幅, 高さ)
            aspect_ratio = max(ellipse[1][0], ellipse[1][1]) / min(ellipse[1][0], ellipse[1][1])
            if min_size[0] <= ellipse[1][0] <= max_size[0] and min_size[1] <= ellipse[1][1] <= max_size[1] and aspect_ratio > 1.6:
                # 楕円の中心座標
                center = ellipse[0]
                # KDTreeを使用して近くの中心を検索
                indices = tree.query_ball_point(center, 20)
                count = len(indices) - 1  # 自分自身を除く
                if count >= 2:
                    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                else:
                    # 垂直および水平の線を引く
                    for other_center in centers:
                        if other_center != center:
                            if abs(other_center[0] - center[0]) < 50 or abs(other_center[1] - center[1]) < 50:
                                cv2.line(image, (int(center[0]), 0), (int(center[0]), image.shape[0]), (255, 0, 0), 1)
                                cv2.line(image, (0, int(center[1])), (image.shape[1], int(center[1])), (255, 0, 0), 1)
                                break

# マルチスレッドで輪郭を処理
with ThreadPoolExecutor() as executor:
    executor.map(process_contour, contours)

# 結果を表示
cv2.imshow('Detected Objects', image)
cv2.imwrite('out.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()