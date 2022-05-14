from func_gdal import *
from func_elements import *


class Villages:
    def __init__(self, path, remote_size=(2560, 2560), dem_size=(160, 160)):
        self.path = path
        self.remote_size = remote_size
        self.dem_size = dem_size
        self.sv_path = self.path.parent / 'traditional villages'
        self.size_vary = self.sv_path / 'sizeVary'
        self.img_dem = self.sv_path / 'DEMImages'

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

    def villages_info(self):
        remote_path = self.sv_path / 'remote'
        name_list = []
        for fileName in remote_path.iterdir():
            name_list.append(fileName.stem[:-7])
        names = np.array(name_list)
        np.savetxt(str(self.sv_path / 'villages_info.csv'), names, fmt='%s')
