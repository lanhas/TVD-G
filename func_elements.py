import json
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from func_pillow import *
from labelme import utils
from skimage import measure
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

            utils.lblsave(str(result_path / (fileName.stem + ".png")), new)
            print("Saved " + str(fileName).split('.')[0] + ".png")


def element_extract(ori_path, elements_name, elements_num, res_path):
    """
    将mask中的地物要素进行提取，分别保存到指定目录

    Paraments
    ---------
    ori_path: Path
        待提取数据根目录
    elements_name: tuple
        需提取的要素名
    elements_num: tuple
        需提取的要素值
    res_path: Path
        结果的保存路径
    """
    for fileName in ori_path.iterdir():
        image = np.array(Image.open(fileName), np.uint8)
        for element_index, element_name in enumerate(elements_name.iterdir()):
            res_image = np.zeros_like(image, np.uint8)
            res_image[image==elements_num[element_index]] = 1
            utils.lblsave(res_path / elements_name[element_index] / fileName.name, res_image)
            print("Element:{} saved in {}".format(element_name, res_path / elements_name[element_index]/ fileName.name))


def element_merge(ori_path, elements_name, elements_num, res_path):
    """
    将多个要素合并起来并保存为标签结果
    Parameters
    ----------
    ori_path: tuple
        各要素的根目录
    elements_name: tuple
        需提取的要素名, 需按重要性照顺序进行排序，越靠前越容易被覆盖
    elements_num: tuple
        需提取的要素值
    result_path: Path
        结果路径
    """
    for file_name in (ori_path / elements_name[0]).iterdir():
        temp_image = np.array(Image.open(file_name), np.uint8)  # 第一个要素的掩模图
        res_image = np.zeros_like(temp_image, np.uint8)
        res_image[temp_image==1] = elements_num[0]               # 写入第一个要素
        # 写入剩余要素
        for element_index, element_name in enumerate(elements_name[1:]):
            fileName = ori_path / element_name / file_name.name
            image = np.array(Image.open(fileName), np.uint8)
            res_image[image==1] = elements_num[element_index+1]
        # 填充剩余要素
        res_image[res_image==0] = 2
        # 保存合并结果
        utils.lblsave(res_path / file_name.name, res_image)
        print("Saved Segmentation data {} in {}".format(file_name.name, res_path / file_name.name))


def element_adjust(source_path, target_path):
    """
    对要素进行调整，去除边界毛糙部分和小区域
    """
    for _, fileName in enumerate(source_path.iterdir()):
        name = fileName.name
        print(name)
        image = np.array(Image.open(fileName), np.uint8)
        image = boundary_pruning(image, 10, 2)
        utils.lblsave(target_path / name, image)


def get_contourline(ori_path, res_path):
    """
    根据DEM图获得山脚线
    """
    for fileName in ori_path.iterdir():
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
        res_image.save(res_path / fileName.name)
        print("Saved contourLine data {} in {}".format(fileName.stem, res_path / fileName.name))


def get_plain(ori_path, res_path):
    """
    优化山脚线得到山体、平原区域
    """
    for fileName in ori_path.iterdir():
        print(fileName.name)
        image = np.array(Image.open(fileName))
        kernel = np.ones((5, 5), np.uint8)
        # 中值滤波平滑边缘
        image = cv2.medianBlur(image, 17)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
        image = Image.fromarray(image)
        image.save(res_path / fileName.name, quantile=95)
        print("Saved contourLine data {} in {}".format(fileName.stem, res_path / fileName.name))


def split_altitute(ori_path, res_path):
    """
    根据高度值对dem进行划分
    """
    for fileName in ori_path.iterdir():
        # print(fileName.name)
        image = np.array(Image.open(fileName))
        print("{}: ({}, {})".format(fileName.stem, np.min(image).astype(int), np.max(image).astype(int)))
        res = np.zeros_like(image, dtype=np.uint8)
        res[image > 500] = 255
        res = Image.fromarray(res)
        res.save(res_path / (fileName.stem + '.jpg'), quantile=95)


def change_name(path, separator, suffix):
    """
    更改路径下文件的名字
    Parameters
    ----------
    path: pathlib.Path
        文件夹路径
    separator: str
        需要分隔的符号，example: '_'
    suffix: str
        新文件的后缀，example: '.jpg'
    """
    for _, fileName in enumerate(path.iterdir()):
        name = fileName.name
        newName = path / (name.split(separator)[0] + suffix)
        fileName.rename(newName)


def delete_file(path, nameList):
    """
    递归删除文件
    Parameters
    ----------
    path: pathlib.Path
        文件夹路径
    nameList: list
        需要删除的文件名列表，example：['baiba', 'bagui']
    """
    for fileName in path.iterdir():
        if fileName.is_dir():
            delete_file(fileName, nameList)
        else:
            if fileName.name.split('.')[0] in nameList:
                fileName.unlink()
                print('delete file {}'.format(fileName.name))


def use_json_to_dataset():
    json_path = Path.cwd() / "data/process/mask/village"
    result_path = Path.cwd() / "data/result/village"
    classes = ["_background_", "village", "wasteland", "unknow"]
    json2dataset(json_path, result_path, classes)


def use_element_extract():
    ori_path = Path.cwd() / "data/result"
    elements_name = ("forest", "farm", "mountain", "water", "village")
    elements_num = (2, 3, 1, 4, 6)
    sv_path = Path.cwd() / "data/result/temp"
    element_merge(ori_path, elements_name, elements_num, sv_path)


def use_element_merge():
    ori_path = Path.cwd() / "data/result"
    elements_name = ("forest", "farm", "mountain", "water", "village")
    elements_num = (2, 3, 1, 4, 6)
    sv_path = Path.cwd() / "data/result/temp"
    element_merge(ori_path, elements_name, elements_num, sv_path)


def use_generate_file():
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


def road_extract(ori_path):
    train_path = ori_path / 'train'
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
            newPath_sat = ori_path / 'valid' / (fileName + '_sat.jpg')
            newPath_mask = ori_path / 'valid' / (fileName + '_mask.png')
            newPath_sat.write_bytes((train_path / (fileName + '_sat.jpg')).read_bytes())
            newPath_mask.write_bytes((train_path / (fileName + '_mask.png')).read_bytes())
            (train_path / (fileName + '_sat.jpg')).unlink()
            (train_path / (fileName + '_mask.png')).unlink()
    else:
        print("error!")


if __name__ == "__main__":
    root_path = Path(r'F:\dataset\villageLand\data_2')
    ori_path = root_path / 'original' / 'dem'
    res_path = root_path / 'process' / 'elements' / 'bmp'
    # res_path = root_path / 'process' / 'elements' / 'mountain'
    from main import tif2bmp_dir

    tif2bmp_dir(ori_path, res_path)
    # get_contourline(ori_path, res_path)

    # ori_path = Path.cwd() / "data/result"
    # elements_name = ("forest", "farm", "mountain", "water", "village")
    # elements_num = (2, 3, 1, 4, 6)
    # res_path = Path.cwd() / "data/result/temp"
    # element_merge(ori_path, elements_name, elements_num, res_path)
