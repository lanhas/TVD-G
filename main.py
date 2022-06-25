from pathlib import Path
from villages import Villages


if __name__ == "__main__":
    data_path = Path(r'F:\Dataset\village data\ori_QDN')
    sv_path = Path(r'F:\Dataset\traditional villages_QDN')
    vil = Villages(data_path, sv_path)
    # vil.run()
    # vil.draw_topoMap()
    vil.concat()

    # # step1: 提取数据
    # vil.separate()
    #
    # # step2: 数据规整
    # vil.clean()
    # # vil.crop()
    # # vil.locNorm()
    #
    # # step3: 对DEM数据进行渲染
    # vil.render_dem()
    #
    # # step4: 绘制等高线
    # vil.draw_contourLine()
    #
    # # step4: 生成标记文件
    # vil.villages_info()

    # vil.draw_topoMap()

    # from PIL import Image
    # import numpy as np
    # from utils import tif2bmp
    # data = r'C:\Users\nscn\Desktop\village paper\示例数据\tif data'
    # for fileName in Path(data).iterdir():
    #     if fileName.stem[-3:] == 'dem':
    #         image = Image.open(fileName)
    #         image = tif2bmp(image)
    #         image = image.resize((1024, 1024))
    #         sv_path = fileName.parent.parent / (fileName.stem + '.png')
    #         image.save(sv_path)
    #     if fileName.stem[-6:] == 'remote':
    #         image = Image.open(fileName)
    #         image = image.resize((1024, 1024))
    #         sv_path = fileName.parent.parent / (fileName.stem + '.png')
    #         image.save(sv_path)
    # # img = np.array(image)
    # # image = img

