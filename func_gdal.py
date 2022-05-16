import math
import numpy as np
from osgeo import gdal
from pathlib import Path
from PIL import Image


def separate_data(path, sv_path=None):
    """
    分离文件，将下载后的混合文件分离
    Parameters
    ----------
    path: Path
        从91卫图下载的原始文件
    sv_path: Path
        待保存的目录
    """
    source_path = path

    def separate(source, target):
        if not source.is_dir():
            return
        for fileName in source.iterdir():
            if fileName.name.split('_')[-1] == '影像.tif':
                newPath = target / "remote" / (fileName.name.split('_影像')[0] + '_remote.tif')
                if not newPath.exists():
                    newPath.write_bytes(fileName.read_bytes())
                    print("remote file {} has been copyed to target folder".format(fileName.name.split('_影像')[0]))
            elif fileName.name.split('_')[-1] == '高程.tif':
                newPath = target / "dem" / (fileName.name.split('_高程')[0] + '_dem.tif')
                if not newPath.exists():
                    newPath.write_bytes(fileName.read_bytes())
                    print("dem file {} has been copyed to target folder".format(fileName.name.split('_高程')[0]))
            else:
                # 递归查找
                separate(fileName, target)
    if sv_path is None:
        sv_path = source_path.parent / 'traditional villages'
    sv_path.mkdir(exist_ok=True)
    (sv_path / 'remote').mkdir(exist_ok=True)
    (sv_path / 'dem').mkdir(exist_ok=True)
    separate(source_path, sv_path)


def clean_data(path, remote_size=(2560, 2560), dem_size=(160, 160), sv_path=None):
    """
    数据对齐，得到大小相同的一组数据
    """
    source_path = path
    # step1: 文件对齐，删除缺少另一组模态文件的数据
    path_remote = source_path / "remote"
    path_dem = source_path / "dem"
    remote_list = [fileName.stem[:-7] for fileName in path_remote.iterdir()]
    dem_list = [fileName.stem[:-4] for fileName in path_dem.iterdir()]
    # 找出共同存在的文件
    common_list = [village for village in remote_list if village in dem_list]
    # 获取冗余数据
    remote_excess = [village for village in remote_list if village not in common_list]
    dem_excess = [village for village in dem_list if village not in common_list]
    # 删除冗余数据
    for village in remote_excess:
        (path_remote / (village + '_remote.tif')).unlink()
        print("Deleted remote data {}".format((village + '_remote.tif')))
    for village in dem_excess:
        (path_dem / (village + '_dem.tif')).unlink()
        print("Deleted dem data {}".format((village + '_remote.tif')))

    # step2: 数据对齐，将不满足data_size大小的数据移动到目标文件夹
    if sv_path is None:
        sv_path = source_path / 'sizeVary'
    output_remote = sv_path / 'remote'
    output_dem = sv_path / 'dem'
    sv_path.mkdir(exist_ok=True)
    output_remote.mkdir(exist_ok=True)
    output_dem.mkdir(exist_ok=True)
    unlink_list = []
    for village_remote, village_dem in zip(path_remote.iterdir(), path_dem.iterdir()):
        img_remote = Image.open(village_remote)
        img_dem = Image.open(village_dem)
        if (img_remote.width, img_remote.height) != remote_size or (img_dem.width, img_dem.height) != dem_size:
            new_remote = output_remote / village_remote.name
            new_dem = output_dem / village_dem.name
            # 移动遥感文件
            if not new_remote.exists():
                with new_remote.open(mode='xb') as fid:
                    fid.write(village_remote.read_bytes())
                unlink_list.append(village_remote)
                print("village {} has been removed in {} and {}".format(village_remote.name, new_remote, new_dem))
            else:
                print("File {} exist! please examine!".format(new_remote))
            # 移动DEM文件
            if not new_dem.exists():
                with new_dem.open(mode='xb') as fid:
                    fid.write(village_dem.read_bytes())
                unlink_list.append(village_dem)
                print("village {} has been removed in {} and {}".format(village_remote.name, new_remote, new_dem))
            else:
                print("File {} exist! please examine!".format(new_remote))
    # 删除已复制的文件
    for fileName in unlink_list:
        fileName.unlink()

    print('Data cleaning completed!')


def crop_data(path):
    """
    将dem数据与遥感数据裁剪到固定大小（同box大小相同)
    数据类型:
        dem：.tif格式，大小152*152, 168*151等, 精度8.5米/像素
        遥感数据：.tif格式，大小为2560*2560， 2304*2304, 2304*2560等， 精度0.53米/像素
    Parameters
    ----------
    ori_path: Path
        数据根目录
    sv_path: Path
        目标根目录
    box_remote: tuple
        遥感数据目标大小{(box_width, box_height)}
    """
    source_path = path / 'remote'
    from PIL import Image
    for fileName in source_path.iterdir():
        image = Image.open(fileName)
        if image.width != 2560 or image.height != 2560:
            print("village {}, width={}, height={}".format(fileName.stem, image.width, image.height))
        # # 读取tif文件
        # remote_data, remote_geotrans, remote_width, remote_height, remote_proj = read_tif(fileName)
        # dem_data, dem_geotrans, dem_width, dem_height, dem_proj = read_tif(ori_path / "dem" / fileName.name)
        # # 根据遥感数据box计算dem box大小
        # dem_width = round(box_remote[0] * (remote_geotrans[1] / dem_geotrans[1]))
        # dem_height = round(box_remote[1] * (remote_geotrans[5] / dem_geotrans[5]))
        # box_dem = (dem_width, dem_height)
        # # 裁剪数据
        # remote_data = remote_data[:, :box_remote[1], :box_remote[0]]
        # dem_data = dem_data[:box_dem[1], :box_dem[0]]
        # # 保存数据
        # remote_width = box_remote[0]
        # remote_height = box_remote[1]
        # dem_width = box_dem[0]
        # dem_height = box_dem[1]
        # res_romote = (remote_data, remote_geotrans, remote_width, remote_height, remote_proj)
        # res_dem = (dem_data, dem_geotrans, dem_width, dem_height, dem_proj)
        # save_tif(res_romote, sv_path / "remote" / fileName.name)
        # print("Saved croped remote data {} in {}".format(fileName.stem, sv_path / "remote" / fileName.name))
        # save_tif(res_dem, sv_path / "dem" / fileName.name)
        # print("Saved croped dem data {} in {}".format(fileName.stem, sv_path / "dem" / fileName.name))


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


def locNorm_dir(path, sv_path=None):
    """
    将文件夹内的遥感和dem数据进行批量对其
    ori_path: Path
        待提取数据根目录
    sv_path: Path
        可选参数，结果的保存路径
    """

    source_path = path
    if sv_path is None:
        sv_path = source_path / 'locNorm'
    sv_path.mkdir(exist_ok=True)
    (sv_path / 'remote').mkdir(exist_ok=True)
    (sv_path / 'dem').mkdir(exist_ok=True)
    remote_dir = path / "remote"
    dem_dir = path / "dem"

    for fileName_remote in remote_dir.iterdir():
        fileName_dem = dem_dir / (fileName_remote.stem[:-7] + '_dem.tif')
        data_remote, data_dem = locNorm(fileName_remote, fileName_dem)
        save_tif(data_remote, sv_path / "remote" / fileName_remote.name)
        print("Saved remote data {} in {}".format(fileName_remote.stem, sv_path / "remote" / fileName_remote.name))
        save_tif(data_dem, sv_path / "dem" / fileName_dem.name)
        print("Saved dem data {} in {}".format(fileName_dem.stem, sv_path / "dem" / fileName_remote.name))


def locNorm(remotePath, demPath):
    """
    数据对齐，将遥感数据和DEM数据按照经纬度对其
    对齐方式:取交集，裁剪非交集部分

    Parameters
    ----------
    remotePath:Path
        遥感数据路径
    demPath:Path
        DEM数据路径

    Results
    -------
    result_remote:tuple
        tif文件格式，包括(im_data, im_geotrans, im_width, im_height, im_proj)
    result_dem
        tif文件格式，包括(im_data, im_geotrans, im_width, im_height, im_proj)
    """

    if remotePath.name != demPath.name:
        raise ValueError("data error!")
    remote_data, remote_geotrans, remote_width, remote_height, remote_proj = read_tif(remotePath)  # 加载遥感数据
    dem_data, dem_geotrans, dem_width, dem_height, dem_proj = read_tif(demPath)   # 加载高程数据
    remote_lt = (remote_geotrans[0], remote_geotrans[3])    # 遥感数据左上角经纬度
    dem_lt = (dem_geotrans[0], dem_geotrans[3])             # 高程数据左上角经纬度
    remote_grid = (remote_geotrans[1], remote_geotrans[5]) # 遥感数据经纬度格网（经度， 纬度）
    dem_grid = (dem_geotrans[1], dem_geotrans[5])          # 高程数据经纬度格网（经度， 纬度）

    # 获取遥感数据和高程数据右下角经纬度坐标
    remote_rb = coordinateTransform((remote_width, remote_height), remote_lt, remote_grid, 'p2l')    # 遥感数据右下角经纬度
    dem_rb = coordinateTransform((dem_width, dem_height), dem_lt, dem_grid, 'p2l')                   # 高程数据右下角经纬度
    # 获取最大范围（遥感U高程）
    union_lt = (min(remote_lt[0], dem_lt[0]), max(remote_lt[1], dem_lt[1]))       # 左上角
    union_rb = (max(remote_rb[0], dem_rb[0]), min(remote_rb[1], dem_rb[1]))       # 右下角
    # 将经纬度坐标转换为图像像素点坐标
    remote_pixel_lt = coordinateTransform(remote_lt, union_lt, remote_grid, mode='l2p')
    remote_pixel_rb = coordinateTransform(remote_rb, union_lt, remote_grid, mode='l2p')
    dem_pixel_lt = coordinateTransform(dem_lt, union_lt, remote_grid, mode='l2p')
    dem_pixel_rb = coordinateTransform(dem_rb, union_lt, remote_grid, mode='l2p')
    union_pixel_rb = coordinateTransform(union_rb, union_lt, remote_grid, mode='l2p')
    # 构造遥感数据和高程数据的缓存文件
    union_remote = np.zeros((union_pixel_rb[1], union_pixel_rb[0]))
    union_dem = union_remote.copy()
    # 将两个缓存文件中包含各自数据的地方置为1
    union_remote[remote_pixel_lt[1]:remote_pixel_rb[1], remote_pixel_lt[0]:remote_pixel_rb[0]] = 1
    union_dem[dem_pixel_lt[1]:dem_pixel_rb[1], dem_pixel_lt[0]:dem_pixel_rb[0]] = 1
    union_data = union_remote + union_dem
    # 获取高程数据和遥感数据的交集
    intersection_data = np.where(union_data==2)
    # 获取交集区域左上角和右下角的坐标
    fin_pixel_lt = (intersection_data[1][0], intersection_data[0][0])
    fin_pixel_rb = (intersection_data[1][-1], intersection_data[0][-1])
    # 将交集区域转换为各自的相对坐标
    relative_remote_lt = (fin_pixel_lt[0] - remote_pixel_lt[0], fin_pixel_lt[1] - remote_pixel_lt[1])
    relative_remote_rb = (fin_pixel_rb[0] - remote_pixel_lt[0], fin_pixel_rb[1] - remote_pixel_lt[1])

    relative_dem_lt = (fin_pixel_lt[0] - dem_pixel_lt[0], fin_pixel_lt[1] - dem_pixel_lt[1])
    relative_dem_rb = (fin_pixel_rb[0] - dem_pixel_lt[0], fin_pixel_rb[1] - dem_pixel_lt[1])
    # 获取dem数据的相对位置
    dem_lt_temp = coordinateTransform(relative_dem_lt, dem_lt, remote_grid, 'p2l')
    dem_rb_temp = coordinateTransform(relative_dem_rb, dem_lt, remote_grid, 'p2l')
    relative_dem_lt = coordinateTransform(dem_lt_temp, dem_lt, dem_grid, 'l2p')
    relative_dem_rb = coordinateTransform(dem_rb_temp, dem_lt, dem_grid, 'l2p')
    # 裁剪图像
    remote_data = remote_data[:, relative_remote_lt[1]:relative_remote_rb[1], relative_remote_lt[0]:relative_remote_rb[0]]
    dem_data = dem_data[relative_dem_lt[1]:relative_dem_rb[1], relative_dem_lt[0]:relative_dem_rb[0]]
    # 获取遥感数据和高程数据左上角的经纬度
    remote_point = coordinateTransform(relative_remote_lt, remote_lt, remote_grid, 'p2l')
    dem_point = dem_lt_temp
    # 更新geotrans
    remote_geotrans = list(remote_geotrans)
    dem_geotrans = list(dem_geotrans)
    remote_geotrans[0] = remote_point[0]
    remote_geotrans[3] = remote_point[1]
    dem_geotrans[0] = dem_point[0]
    dem_geotrans[3] = dem_point[1]
    remote_geotrans = tuple(remote_geotrans)
    dem_geotrans = tuple(dem_geotrans)
    # 更新图像宽高
    remote_width = remote_data.shape[2]
    remote_height = remote_data.shape[1]
    dem_width = dem_data.shape[1]
    dem_height = dem_data.shape[0]
    # 构造结果
    result_remote = (remote_data, remote_geotrans, remote_width, remote_height, remote_proj)
    result_dem = (dem_data, dem_geotrans, dem_width, dem_height, dem_proj)
    return result_remote, result_dem


def read_tif(filePath, mode=0):
    """
    读取tif文件
    Parameters
    ----------
    filePath: str
        文件路径

    Return
    ------
    im_data: ndarray
        图像数据
    im_geotrans: tuple
        仿射矩阵，共有六个参数，依次为{左上角x坐标；
                                    东西方向上图像的分辨率；
                                    地图的旋转角度，0表示图像的行与x轴平行；
                                    左上角y坐标；
                                    地图的旋转角度，0表示图像的列与y轴平行；
                                    南北方向上地图的分辨率}
    im_width: int
    im_height: int
    im_proj: 地图投影信息
    """
    data = gdal.Open(str(filePath))
    im_width = data.RasterXSize  # 读取图像的宽度，x方向上的像素个数
    im_height = data.RasterYSize  # 读取图像的高度，y方向上的像素个数
    im_geotrans = data.GetGeoTransform()  # 仿射矩阵
    im_proj = data.GetProjection()  # 地图投影信息
    im_data = data.ReadAsArray(0, 0, im_width, im_height).astype(np.float64)  # 将数据写成数组，对应栅格矩阵
    del data  # 关闭对象，文件dataset
    if mode == 0:
        return im_data, im_geotrans, im_width, im_height, im_proj
    elif mode == 1:
        return im_data
    else:
        raise ValueError("mode error!")


def save_tif(data, filePath):
    """
    将tif格式数据保存为tif文件

    Parameters:
    data: tuple
        tif格式数据，包括(im_data, im_geotrans, im_width, im_height, im_proj)
    filePath: Path
        文件路径
    """
    im_data, im_geotrans, im_width, im_height, im_proj = data
    if im_data is None:
        raise ValueError("Data is None!")
    if len(im_data.shape) == 2:
        channels = 1
        data_type = gdal.GDT_Float32
    elif len(im_data.shape) == 3:
        channels = 3
        data_type = gdal.GDT_Byte
    else:
        raise ValueError("data error!")
    driver = gdal.GetDriverByName("GTiff")
    data = driver.Create(str(filePath), im_width, im_height, channels, data_type)
    data.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    data.SetProjection(im_proj)  # 写入投影
    if channels == 3:
        for i in range(channels):
            data.GetRasterBand(i + 1).WriteArray(im_data[i])
    else:
        data.GetRasterBand(1).WriteArray(im_data)
    del data


def coordinateTransform(data, stPoint, grad, mode='l2p'):
    """
    坐标转换, 经纬度坐标和像素点坐标之间的转换
    Parameters
    ----------
    data: tuple
        代转换的数据
    stPoint: tuple
        参考点的经纬度坐标，即左上角点的经纬度坐标
    grad: tuple
        格网宽度
    mode: str
        转换模式{l2p: 经纬度转像素点
                p2l: 像素点转经纬度}
    Return
    ------
    result:tuple
    """
    if mode == 'l2p':
        # 将经纬度坐标转为相对坐标
        px = math.ceil((data[0] - stPoint[0]) / grad[0])
        py = math.ceil((data[1] - stPoint[1]) / grad[1])
    elif mode == 'p2l':
        # 将相对坐标转为经纬度坐标
        px = stPoint[0] + data[0] * grad[0]
        py = stPoint[1] + data[1] * grad[1]
    else:
        raise ValueError('mode error!')
    return [px, py]


if __name__ == "__main__":
    ori_path = Path(r'F:\Dataset\villageLand\data_team\team2_data')    # root dir
    # clean_data(ori_path)
    # splited_path is Optional parameters
    # splited_path = Path(r'G:\dataset\temp_splited')
    # separate_data(ori_path)
    locNorm_dir(ori_path)


