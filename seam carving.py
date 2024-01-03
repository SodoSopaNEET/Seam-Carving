import cv2
import numpy as np

image = cv2.imread("image.png")  # 讀取影像

height, width, _ = image.shape
delete_mask = np.zeros((height, width))
x, y, w, h = cv2.selectROI("img", image)
delete_mask[y:y + h, x:x + w] = 10000

# 執行多次(例如執行100次則會刪除100點寬度)
for i in range(100):

    # 影像梯度

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 將影像轉為灰階

    gradient_y = cv2.Sobel(image_gray, cv2.CV_64F, dx=0, dy=1)  # 計算Y軸方向的梯度
    gradient_x = cv2.Sobel(image_gray, cv2.CV_64F, dx=1, dy=0)  # 計算X軸方向的梯度

    gradient_image = np.hypot(gradient_x, gradient_y)  # hypot = x平方+y平方

    # gradient_image = gradient_image / gradient_image.max()  # 將圖片變成可以顯示的格式

    gradient_image = np.subtract(gradient_image, delete_mask)

    # cv2.imshow("gradient img", gradient_image)  # 顯示「梯度影像」
    # cv2.waitKey(0)  # 開啟影像直到按下任意鍵
    # cv2.destroyAllWindows()  # 關閉影像視窗

    # 能量圖

    height, width = gradient_image.shape  # 取得影像的高度與寬度
    energy_map = gradient_image.copy()  # 複製一張與原始影像一樣的圖片，用來處理

    # 利用迴圈(loop)掃描影像中的每個值
    for j in range(height):
        for i in range(width):

            # 假如處在影像第一列(row)，不做任何處理
            if j == 0:
                continue

            # 假如在影像最左邊的處理方式
            if i == 0:
                energy_map[j, i] = (energy_map[j, i]) + min((energy_map[j - 1, i]), (energy_map[j - 1, i + 1]))

            # 假如在影像最右邊的處理方式
            elif i == width - 1:
                energy_map[j, i] = (energy_map[j, i]) + min((energy_map[j - 1, i - 1]), (energy_map[j - 1, i]))

            # 一般處理方式
            else:
                energy_map[j, i] = (energy_map[j, i]) + min((energy_map[j - 1, i - 1]), (energy_map[j - 1, i]),
                                                            (energy_map[j - 1, i + 1]))

    energy_map = energy_map / np.max(energy_map)  # 將圖片變成可以顯示的格式
    # cv2.imshow('energy map',energy_map)  # 顯示「能量圖」
    # cv2.waitKey(0)  # 開啟影像直到按下任意鍵
    # cv2.destroyAllWindows()  # 關閉影像視窗

    # 反向追蹤

    index_list = [None] * height  # 建立一個列表，用來儲存每列(row)最小值的「位置」

    index = np.argmin(energy_map[-1, :])  # 找出影像中最下方那列(row)的最小值
    index_list[height - 1] = index  # 將最下方那列(row)的最小值位置儲存在列表內

    # 利用迴圈(loop)從下方列開始搜尋路徑，找到每列經過的位置
    for j in range(height - 2, -1, -1):

        # 如果上一個位置在影像最左邊的處理方式
        if index == 0:
            index_list[j] = index + np.argmin(energy_map[j, index:index + 2])
            index = index_list[j]  # 更新位置

        # 如果上一個位置在影像最右邊的處理方式
        elif index == width - 1:
            index_list[j] = index + np.argmin(energy_map[j, index - 1:index + 1]) - 1
            index = index_list[j]  # 更新位置

        # 一般處理方式
        else:
            index_list[j] = index + np.argmin(energy_map[j, index - 1:index + 2]) - 1
            index = index_list[j]  # 更新位置

    # 複製一張原始影像用來畫圖
    image_draw = image.copy()

    # 利用迴圈(loop)將每列(row)要刪除的位置標上紅色
    for j in range(height):
        image_draw[j, index_list[j], :] = [0, 0, 255]  # [Blue, Green, Red]的值

    cv2.imshow('img', image_draw)  # 顯示影像
    cv2.waitKey(50)  # 開啟影像直到按下任意鍵
    # cv2.destroyAllWindows()  # 關閉影像視窗

    # 刪除畫紅線的位置

    new_width = width - 1  # 新的寬度為舊的寬度減1 (一列移除1個位置)
    new_img = np.zeros((height, new_width, 3), dtype=np.uint8)  # 建立一張空白影像
    new_delete_mask = np.zeros((height, new_width))

    for j in range(height):
        new_img[j, :, :] = np.concatenate((image[j, 0:index_list[j]], image[j, index_list[j] + 1:new_width + 1]))
        new_delete_mask[j, :] = np.concatenate(
            (delete_mask[j, 0:index_list[j]], delete_mask[j, index_list[j] + 1:new_width + 1]))

    cv2.imshow('img', new_img)  # 顯示影像
    cv2.waitKey(50)  # 影像顯示50毫秒

    image = new_img  # 更新影像
    width = new_width  # 更新寬度
    delete_mask = new_delete_mask

cv2.waitKey(0)
cv2.destroyAllWindows()  # 關閉影像視窗
