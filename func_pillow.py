import cv2
import numpy as np
from PIL import Image
from labelme import utils


def boundary_pruning(image, kernel_size, iter_num):
    """
    对图像进行开闭运算，去除细小的毛躁区域
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iter_num)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iter_num)
    return image


def image_blend(image, areaMask, alpha, beta, gamma) -> Image:
    """
    将图片内的前背景按一定比例区分

    Parameters
    ----------
    image: ndarray
    areaMask: ndarray
        区域掩膜
    alpha: float
        前景区的融合比例
    beta: float
        背景区的融合比例
    gamma: 透明度

    Return
    ------
    result: ndarray
        融合后的结果
    """
    foreground = image
    background = image.copy()
    # 如果掩膜是单通道图像，先将其转为三通道
    if len(areaMask.shape) == 2:
        for i in range(3):
            foreground[:, :, i][areaMask == 0] = 0
            background[:, :, i][areaMask > 0] = 0
    result = cv2.addWeighted(foreground, alpha, background, beta, gamma)
    return result


def color2annotation(image) -> np.array:
    """
    将三通道的颜色label转为单通道的annotation

    Parameters
    ----------
    image: ndarray

    Return
    ------
    annotation: ndarray
    """
    annotation = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    annotation[cv2.inRange(image, colorMask['urban_land'], colorMask['urban_land'])==255 ] = 6
    annotation[cv2.inRange(image, colorMask['agriculture_land'], colorMask['agriculture_land'])==255] = 3
    annotation[cv2.inRange(image, colorMask['rangeland'], colorMask['rangeland'])==255] = 5
    annotation[cv2.inRange(image, colorMask['forest_land'], colorMask['forest_land'])==255] = 2
    annotation[cv2.inRange(image, colorMask['water'], colorMask['water'])==255] = 4
    annotation[cv2.inRange(image, colorMask['barren_land'], colorMask['barren_land'])==255] = 1
    annotation[cv2.inRange(image, colorMask['unkonwn'], colorMask['unkonwn'])==255] = 0
    return annotation


def getAreaMask(image, areaIndex):
    """
    根据类别标签，获得annotation对应的二值掩膜，0代表背景，1代表该区域

    Parameters
    ----------
    image: ndarray
        图像的label，三通道彩色图像
    areaIndex: int
        区域标签1~类别总数
        urban_land:         1
        agriculture_land:   2
        rangeland:          3
        forest_land:        4
        water:              5
        barren_land:        6
        unkonwn:            7
    Return
    ------
    result: ndarray
        二值掩膜，单通道，0代表背景，1代表该区域
    """
    # 三通道彩色label转单通道annotation
    annotation = color2annotation(image)
    result = np.zeros_like(annotation, dtype=np.uint8)
    if areaIndex == 0:
        result = image
    elif areaIndex == 1:
        result[annotation == 1] = 1
    elif areaIndex == 2:
        result[annotation == 2] = 1
    elif areaIndex == 3:
        result[annotation == 3] = 1
    elif areaIndex == 4:
        result[annotation == 4] = 1
    elif areaIndex == 5:
        result[annotation == 5] = 1
    elif areaIndex == 6:
        result[annotation == 6] = 1
    elif areaIndex == 7:
        result[annotation == 7] = 1
    return result


def img_addition(image, areaMask, axisColor):
    """
    为图片内掩膜区域上色

    Parameters
    ----------
    image: ndarray
    areaMask: ndarray
        区域掩膜
    axisColor: tuple
        颜色，(r, g, b)

    Return
    ------
    image: ndarray
    """
    image[:, :, 0][areaMask > 0] = axisColor[0]
    image[:, :, 1][areaMask > 0] = axisColor[1]
    image[:, :, 2][areaMask > 0] = axisColor[2]
    return image


def convert(path, sv_path):
    """
    转换文件为统一格式
    """
    for fileName in path.iterdir():
        print(fileName.name)
        image = np.array(Image.open(fileName).convert("1"))
        res = np.zeros_like(image, dtype=np.uint8)
        res[image > 0] = 1
        utils.lblsave(sv_path / (fileName.stem + '.png'), res)


colorMask = {'urban_land': (0, 128, 128),
             'agriculture_land': (128, 128, 0),
             'rangeland': (128, 0, 128),
             'forest_land': (0, 128, 0),
             'water': (0, 0, 128),
             'barren_land': (128, 0, 0),
             'unkonwn': (0, 0, 0),
            }