from pathlib import Path
from villages import Villages


if __name__ == "__main__":
    data_path = Path(r'F:\Dataset\village data\ori_GZ')
    vil = Villages(data_path)

    # step1: 提取数据
    vil.separate()

    # step2: 数据规整
    vil.clean()
    vil.crop()
    vil.locNorm()

    # step3: 对DEM数据进行渲染
    vil.render_dem()

    # step4: 生成标记文件
    vil.villages_info()
