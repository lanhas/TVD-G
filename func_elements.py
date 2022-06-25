import json
import base64
import time
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from func_pillow import *
from labelme import utils
from skimage import measure
import matplotlib.pyplot as plt
'''
此处生成的标签图是8位彩色图，每个像素点的值就是这个像素点所属的种类
'''


def json2dataset(json_path, result_path, classes):
    """
    将labelme标记文件转为segmentation

    Parameters
    ----------
    json_path: Path
        json文件根目录
    result_path: Path
        png文件根目录，将结果图像保存为png格式
    classes: List
        类别{["_background_", "A", "B", "C"]}
    """
    for fileName in json_path.iterdir():
        if fileName.suffix == '.json':
            data = json.load(open(fileName))
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = fileName.parent / data['imagePath']
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)
            # temp
            new[new==0] = 2

            utils.lblsave(str(result_path / (fileName.stem + ".png")), new)
            print("Saved " + str(fileName).split('.')[0] + ".png")


def elements_split(path, elements_name, elements_num, sv_path=None):
    """
    将mask中的地物要素进行提取，分别保存到指定目录

    Paraments
    ---------
    path: Path
        待提取数据根目录
    elements_name: tuple
        需提取的要素名
    elements_num: tuple
        需提取的要素值
    sv_path: Path
        结果的保存路径, 如果未指定，将默认保存在../split
    """
    if sv_path is None:
        sv_path = path.parent / 'split'
    sv_path.mkdir(exist_ok=True)

    for fileName in path.iterdir():
        image = np.array(Image.open(fileName), np.uint8)
        for element_index, element_name in enumerate(elements_name):
            res_image = np.zeros_like(image, np.uint8)
            res_image[image == elements_num[element_index]] = 1
            element_path = sv_path / element_name
            element_path.mkdir(exist_ok=True)
            utils.lblsave(element_path / fileName.name, res_image)
            print("Element:{} saved in {}".format(element_name, sv_path / element_name / fileName.name))


def elements_merge(path, elements_name, elements_num, sv_path=None):
    """
    将多个要素合并起来并保存为标签结果
    Parameters
    ----------
    path: Path
        各要素的根目录
    elements_name: tuple
        需提取的要素名, 需按重要性照顺序进行排序，越靠前越容易被覆盖
    elements_num: tuple
        需提取的要素值
    sv_path: Path
        结果路径, 如果未指定，将默认保存在../merge
    """
    if sv_path is None:
        sv_path = path.parent / 'merge'
    sv_path.mkdir(exist_ok=True)

    for file_name in (path / elements_name[0]).iterdir():
        temp_image = np.array(Image.open(file_name), np.uint8)  # 第一个要素的掩模图
        res_image = np.zeros_like(temp_image, np.uint8)
        res_image[temp_image == 1] = elements_num[0]               # 写入第一个要素
        # 写入剩余要素
        for element_index, element_name in enumerate(elements_name[1:]):
            fileName = path / element_name / file_name.name
            image = np.array(Image.open(fileName), np.uint8)
            res_image[image == 1] = elements_num[element_index+1]
        # 填充剩余要素 将剩余的空隙填充为草地
        res_image[res_image == 0] = 5
        # 保存合并结果
        utils.lblsave(sv_path / file_name.name, res_image)
        print("Saved Segmentation data {} in {}".format(file_name.name, sv_path / file_name.name))


def element_adjust(path, sv_path=None):
    """
    对要素进行调整，去除边界毛糙部分和小区域
    parameters
    ----------
    path: Path

    sv_path: Path

    """
    if sv_path is None:
        sv_path = path.parent / 'adjust'
    sv_path.mkdir(exist_ok=True)

    for _, fileName in enumerate(path.iterdir()):
        name = fileName.name
        print(name)
        image = np.array(Image.open(fileName), np.uint8)
        image = boundary_pruning(image, 10, 2)
        utils.lblsave(sv_path / name, image)


def get_contourline(path, sv_path):
    """
    根据DEM图获得山脚线
    """
    for fileName in path.iterdir():
        print(fileName.name)
        image = np.array(Image.open(fileName))
        res = np.zeros_like(image, dtype=np.uint8)
        image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
        # Construct some test data
        x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
        r = np.sin(np.exp((np.sin(x) ** 3 + np.cos(y) ** 2)))

        # Find contours at a constant value of 0.8
        contours = measure.find_contours(image_norm, 0.3, fully_connected='high', positive_orientation='high')
        contours = sorted(contours, key=lambda x:x.shape, reverse=True)[:5]

        for contour in contours:
            contour = contour.astype(np.int32)
            for point in contour:
                res[point[0], point[1]] = 255
        res_image = Image.fromarray(res)
        res_image.save(sv_path / fileName.name)
        print("Saved contourLine data {} in {}".format(fileName.stem, sv_path / fileName.name))


def get_plain(path, sv_path):
    """
    优化山脚线得到山体、平原区域
    """
    for fileName in path.iterdir():
        print(fileName.name)
        image = np.array(Image.open(fileName))
        kernel = np.ones((5, 5), np.uint8)
        # 中值滤波平滑边缘
        image = cv2.medianBlur(image, 17)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
        image = Image.fromarray(image)
        image.save(sv_path / fileName.name, quantile=95)
        print("Saved contourLine data {} in {}".format(fileName.stem, sv_path / fileName.name))


def split_altitute(path, sv_path):
    """
    根据高度值对dem进行划分
    """
    for fileName in path.iterdir():
        # print(fileName.name)
        image = np.array(Image.open(fileName))
        print("{}: ({}, {})".format(fileName.stem, np.min(image).astype(int), np.max(image).astype(int)))
        res = np.zeros_like(image, dtype=np.uint8)
        res[image > 500] = 255
        res = Image.fromarray(res)
        res.save(sv_path / (fileName.stem + '.jpg'), quantile=95)


def use_json_to_dataset():
    json_path = Path.cwd() / "data/process/mask/village"
    result_path = Path.cwd() / "data/result/village"
    classes = ["_background_", "village", "wasteland", "unknow"]
    json2dataset(json_path, result_path, classes)


def use_elements_split(path, sv_path=None):
    """
    Parameters
    ----------
    path: Path
    sv_path: Path
    """
    source_path = path
    if not sv_path:
        sv_path = source_path.parent / 'elements_split'
    elements_name = ("rangeLand", "forest", "farm", "barrenLand", "water", "village")
    elements_num = (5, 2, 3, 1, 4, 6)
    elements_split(source_path, elements_name, elements_num, sv_path)


def use_elements_merge(path, sv_path):
    """
    Parameters
    ----------
    path: Path
    sv_path: Path
    """
    source_path = path
    if not sv_path:
        sv_path = source_path.parent / 'elements_merge'
    elements_name = ("rangeLand", "forest", "farm", "barrenLand", "water", "village")
    elements_num = (5, 2, 3, 1, 4, 6)
    elements_merge(source_path, elements_name, elements_num, sv_path)


def use_generate_file():
    import shutil
    source_remote = Path.cwd() / 'original/JPEGImages'
    source_dem = Path.cwd() / 'original/DEMImages_jpg'
    source_mask = Path.cwd() / 'optimize/SegmentationClass'

    mw = ['bagui', 'sunlan', 'zengchong']
    alongMountain = ['gaohua', 'dalubian', 'zaikua']
    alongRiver = ['liujiang', 'tangzhai', 'xinzhai']
    plain = ['tangya', 'xiaban', 'xiajiu']
    mountain = ['baiguo', 'baiqiao', 'changzhai']

    villageFile_Name = [mw, alongMountain, alongRiver, plain, mountain]
    villageClasses_Name = ['平原', '山地', '山环水绕', '沿河', '依山']

    def mv_file(villageClass_name, villageFile_names):
        propertyClass = ['遥感', 'DEM', '特征图']
        propertyStem = ['.jpg', '.jpg', '.png']
        sourcePath = [source_remote, source_dem, source_mask]
        for source_path, classes_name, image_stem in zip(sourcePath, propertyClass, propertyStem):
            for village_className, village_fileNames in zip(villageClass_name, villageFile_names):
                target_forder = Path.cwd() / '数据集样例' / village_className / classes_name
                for village_name in village_fileNames:
                    file_name = village_name + image_stem
                    shutil.copy(source_path / file_name, target_forder / file_name)

    mv_file(villageClasses_Name, villageFile_Name)


def road_extract(path):
    from sklearn.model_selection import train_test_split
    train_path = path / 'train'
    sat_list = []
    mask_list = []
    for fileName in train_path.iterdir():
        img_name = fileName.name
        if img_name.split('_')[1] == 'sat.jpg':
            sat_list.append(img_name.split('_')[0])
        elif img_name.split('_')[1] == 'mask.png':
            mask_list.append(img_name.split('_')[0])
        else:
            print('error image! please check!')
    if len(sat_list) == len(mask_list):
        train_list, test_list = train_test_split(sat_list, test_size=0.15, random_state=10)
        for fileName in test_list:
            newPath_sat = path / 'valid' / (fileName + '_sat.jpg')
            newPath_mask = path / 'valid' / (fileName + '_mask.png')
            newPath_sat.write_bytes((train_path / (fileName + '_sat.jpg')).read_bytes())
            newPath_mask.write_bytes((train_path / (fileName + '_mask.png')).read_bytes())
            (train_path / (fileName + '_sat.jpg')).unlink()
            (train_path / (fileName + '_mask.png')).unlink()
    else:
        print("error!")


def contour_line(path, sv_path):
    if sv_path is None:
        sv_path = path.parent / 'contourLine'
    sv_path.mkdir(exist_ok=True)

    for fileName in path.iterdir():
        image = Image.open(fileName)
        image = np.array(image)
        X = np.arange(0, image.shape[0])
        Y = np.arange(0, image.shape[1])
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)

        ctf = plt.contourf(X, Y, image, 25)
        plt.contour(X, Y, image, 25)
        # plt.colorbar(ctf)  # 添加cbar
        plt.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        sv_name = sv_path / (fileName.stem + '.png')
        plt.savefig(sv_name, bbox_inches=extent)


def extract_qdn(path, sv_path):
    if sv_path is None:
        sv_path = path.parent / 'split_'
    sv_path.mkdir(exist_ok=True)

    dir_names = ['remote', 'dem', 'DEMImages']
    dir_suffx = ['_remote.tif', '_dem.tif', '_demi.jpg']

    df_village = pd.read_excel(r'F:\Dataset\traditional village_QDN\villages_info.xlsx')
    village_names = df_village['编号']

    for index, dir_name in enumerate(dir_names):
        for village_name in list(village_names):
            oriPath = path / dir_name / (village_name + dir_suffx[index])
            newPath = sv_path / dir_name / (village_name + dir_suffx[index])
            if not newPath.exists():
                newPath.write_bytes(oriPath.read_bytes())
                print("image {} has been copyed to target folder".format(village_name))


def combine_village(co_path, vl_path, sv_path):
    if sv_path is None:
        sv_path = co_path.parent / 'topoMap'
    sv_path.mkdir(exist_ok=True)

    for co_fileName in co_path.iterdir():
        vl_fileName = vl_path / (co_fileName.stem[:-4] + '_village.png')
        sv_fileName = sv_path / (co_fileName.stem[:-4] + '_topo.png')

        img_co = Image.open(co_fileName).resize((1024, 1024))
        img_vl = Image.open(vl_fileName).resize((1024, 1024))
        image = image_blend(np.array(img_co), np.array(img_vl), 0.7, 1, 0)
        image = Image.fromarray(image)
        image.save(sv_fileName)


def concat_vl(re_path, to_path, sv_path, axis=0):
    if sv_path is None:
        sv_path = re_path.parent / 'topoMap_concat'
    sv_path.mkdir(exist_ok=True)
    for re_fileName in re_path.iterdir():
        to_fileName = to_path / (re_fileName.stem[:-7] + '_topo.png')
        sv_fileName = sv_path / (re_fileName.stem[:-7] + '_concat.png')
        image_re, image_to = Image.open(re_fileName), Image.open(to_fileName).resize((2560, 2560))
        size_re, size_to = image_re.size, image_to.size

        if axis == 0:
            joint = Image.new("RGB", (size_re[0] + size_to[0], size_re[1]))
            loc_re, loc_to = (0, 0), (size_re[0], 0)
        else:
            joint = Image.new("RGB", (size_re[0], size_to[1] + size_re[1]))
            loc_re, loc_to = (0, 0), (0, size_re[1])
        joint.paste(image_re, loc_re)
        joint.paste(image_to, loc_to)
        joint.save(sv_fileName)



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


if __name__ == "__main__":
    # root_path = Path(r'F:\Dataset\traditional villages_GZ')
    # sv_path = Path(r'F:\Dataset\traditional village_QDN')
    # extract_qdn(root_path, sv_path)

    path_co = Path(r'F:\Dataset\traditional village_QDN\contourLine')
    path_vl = Path(r'F:\Dataset\traditional village_QDN\villageMask')
    sv_path = Path(r'F:\Dataset\traditional village_QDN\topoMap')

    combine_village(path_co, path_vl, sv_path)
    # img_c = Image.open(r'F:\Dataset\traditional village_QDN\contourLine\GZ1_024_ZC_dem.png').resize((1024, 1024))
    # img_v = Image.open(r'F:\Dataset\traditional village_QDN\villageMask\GZ1_024_ZC_remote.png').resize((1024, 1024))
    # image = image_blend(np.array(img_c), np.array(img_v), 0.7, 1, 0)
    # image = Image.fromarray(image)
    # image.show()



    # classes = ["_background_", "wasteland", "forest", "farm", "water", "glass", "village"]
    # json2dataset(root_path, sv_path, classes)
    # sv_path = root_path / 'process' / 'elements' / 'mountain'
    # get_contourline(path, sv_path)
    # path = Path.cwd() / "data/result"
    # elements_name = ("forest", "farm", "mountain", "water", "village")
    # elements_num = (2, 3, 1, 4, 6)
    # sv_path = Path.cwd() / "data/result/temp"
    # elements_merge(path, elements_name, elements_num, sv_path)
