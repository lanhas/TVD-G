from pathlib import Path
from villages import Villages


if __name__ == "__main__":
    # data_path = Path(r'F:\Dataset\village data\ori_GZ')
    # vil = Villages(data_path)
    #
    # # step1: 提取数据
    # # vil.separate()
    #
    # # step2: 数据规整
    # # vil.clean()
    # # vil.crop()
    # # vil.locNorm()
    #
    # # step3: 对DEM数据进行渲染
    # # vil.render_dem()
    #
    # # step4: 生成标记文件
    # vil.villages_info()
    from PIL import Image
    import numpy as np

    data = r'F:\Dataset\traditional villages_GZ\remote\GZ1_001_PL_remote.tif'
    image = Image.open(data)
    img = np.array(image)
    image = img

