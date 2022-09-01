import cv2
import math
import numpy as np

def localTranslationWarpFastWithStrength(srcImg, startX, startY, endX, endY, radius, strength):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()


    maskImg = np.zeros(srcImg.shape[:2], np.uint8)
    cv2.circle(maskImg, (startX, startY), math.ceil(radius), (255, 255, 255), -1)

    K0 = 100/strength

    # 计算公式中的|m-c|^2
    ddmc_x = (endX - startX) * (endX - startX)
    ddmc_y = (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape

    mapX = np.vstack([np.arange(W).astype(np.float32).reshape(1, -1)] * H)
    mapY = np.hstack([np.arange(H).astype(np.float32).reshape(-1, 1)] * W)

    distance_x = (mapX - startX) * (mapX - startX)
    distance_y = (mapY - startY) * (mapY - startY)
    distance = distance_x + distance_y
    K1 = np.sqrt(distance)
    ratio_x = (ddradius - distance_x) / (ddradius - distance_x + K0 * ddmc_x)
    ratio_y = (ddradius - distance_y) / (ddradius - distance_y + K0 * ddmc_y)
    ratio_x = ratio_x * ratio_x
    ratio_y = ratio_y * ratio_y

    UX = mapX - ratio_x * (endX - startX) * (1 - K1/radius)
    UY = mapY - ratio_y * (endY - startY) * (1 - K1/radius)

    np.copyto(UX, mapX, where=maskImg == 0)
    np.copyto(UY, mapY, where=maskImg == 0)
    UX = UX.astype(np.float32)
    UY = UY.astype(np.float32)
    copyImg = cv2.remap(srcImg, UX, UY, interpolation=cv2.INTER_LINEAR)

    return copyImg



image = cv2.imread("/Users/anshijie/Downloads/000000.jpeg")
processed_image = image.copy()
startX_left, startY_left, endX_left, endY_left = 101, 266, 192, 233
startX_right, startY_right, endX_right, endY_right = 287, 275, 192, 233
radius = 45
strength = 100
# 瘦左边脸
processed_image = localTranslationWarpFastWithStrength(processed_image, startX_left, startY_left, endX_left, endY_left, radius, strength)
# 瘦右边脸
processed_image = localTranslationWarpFastWithStrength(processed_image, startX_right, startY_right, endX_right, endY_right, radius, strength)
cv2.imwrite("thin.jpg", processed_image)
