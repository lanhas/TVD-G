import numpy as np
from PIL import Image


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


def render_dem(path, sv_path=None):
    source_path = path / 'dem'
    if sv_path is None:
        sv_path = path / 'DEMImages'
    sv_path.mkdir(exist_ok=True)
    for fileName in source_path.iterdir():
        image = tif2bmp(Image.open(fileName))
        image.save(sv_path / (fileName.stem + '.jpg'))
        print("Saved contourLine data {} in {}".format(fileName.stem, sv_path / (fileName.stem + '.jpg')))


def tif2bmp(img: Image)->Image:
    """
    将高程tif文件转为bmp文件，压缩方法：
        255 * (currentNum - minNum)/(maxNum - minNum)
    """
    dem_array = np.array(img)
    dem_min = dem_array.min()
    dem_max = dem_array.max()
    dem_array = 255 * np.divide(dem_array - dem_min, dem_max - dem_min)
    dem_array = dem_array.astype(np.uint8)
    result = Image.fromarray(dem_array)
    return result


# if __name__ == "__main__":
#     from pathlib import Path
#     path = Path(r'F:\Dataset\traditional villages_QDN\villageMask')
#     for fileName in path.iterdir():
#         new_path = fileName.parent / (fileName.stem[:-7] + '_village.png')
#         fileName.rename(new_path)
