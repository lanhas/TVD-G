from func_gdal import *
from func_elements import *
from utils import render_dem


class Villages:
    def __init__(self, path, sv_path=None, remote_size=(2560, 2560), dem_size=(160, 160)):
        self.path = path
        self.remote_size = remote_size
        self.dem_size = dem_size
        if sv_path is None:
            self.sv_path = self.path.parent / 'traditional villages'
        else:
            self.sv_path = sv_path
        self.remote_path = self.sv_path / 'remote'
        self.size_vary = self.sv_path / 'sizeVary'
        self.img_dem = self.sv_path / 'DEMImages'
        self.img_contourLine = self.sv_path / 'contourLine'
        self.img_villageMask = self.sv_path / "villageMask"
        self.img_topoMap = self.sv_path / "topoMap"
        self.img_concat = self.sv_path / "topoMap_concat"

    def separate(self):
        separate_data(self.path, self.sv_path)

    def clean(self):
        clean_data(self.sv_path, self.remote_size, self.dem_size, self.size_vary)

    def crop(self):
        crop_data(self.size_vary)

    def locNorm(self):
        pass
        # locNorm_dir(self.sv_path)

    def render_dem(self):
        render_dem(self.sv_path, self.img_dem)

    def draw_contourLine(self):
        contour_line(self.sv_path / 'dem', self.img_contourLine)

    def villages_info(self):

        name_list = []
        for fileName in self.remote_path.iterdir():
            name_list.append(fileName.stem[:-7])
        names = np.array(name_list)
        np.savetxt(str(self.sv_path / 'villages_info.csv'), names, fmt='%s')

    def draw_topoMap(self):
        if not self.img_villageMask.exists():
            raise ValueError('village Mask not exist!')
        combine_village(self.img_contourLine, self.img_villageMask, self.img_topoMap)

    def concat(self, axis=0):
        concat_vl(self.remote_path, self.img_topoMap, self.img_concat, axis)

    def run(self):

        # step1: 提取数据
        self.separate()

        # step2: 数据规整
        self.clean()
        self.crop()
        self.locNorm()

        # step3: 对DEM数据进行渲染
        self.render_dem()

        # step4: 绘制等高线
        self.draw_contourLine()

        # step5: 生成标记文件
        self.villages_info()
