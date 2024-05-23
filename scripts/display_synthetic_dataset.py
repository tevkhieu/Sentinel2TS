import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from sentinel2_ts.utils.visualize import plot_single_spectral_signature
f = open("C:/Users/tevch/Documents/Stage/usgs_splib07/ASCIIdata/ASCIIdata_splib07b_rsSentinel2/ChapterA_ArtificialMaterials/S07SNTL2_Concrete_WTC01-37A_ASDFRa_AREF.txt")
concrete = f.readlines()[1:]
f.close()
# f = open("C:/Users/tevch/Documents/Stage/usgs_splib07/ASCIIdata/ASCIIdata_splib07b_rsSentinel2/ChapterV_Vegetation/S07SNTL2_Aspen_Aspen-1_green-top_ASDFRa_AREF.txt")
f = open("C:/Users/tevch/Documents/Stage/usgs_splib07/ASCIIdata/ASCIIdata_splib07b_rsSentinel2/ChapterV_Vegetation/S07SNTL2_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt")
vegetation_1 = f.readlines()[1:]
f.close()
f = open("C:/Users/tevch/Documents/Stage/usgs_splib07/ASCIIdata/ASCIIdata_splib07b_rsSentinel2/ChapterV_Vegetation/S07SNTL2_Grass_dry.4+.6green_AMX27_BECKa_AREF.txt")
vegetation_2 = f.readlines()[1:]
f.close()

index = [1, 2, 3, 8, 4, 5, 6, 9, 11, 12]
concrete = np.array([float(i.split()[0]) for i in concrete])[index]
spring_vegetation = np.array([float(i.split()[0]) for i in vegetation_1])[index]
summer_vegetation = np.array([float(i.split()[0]) for i in vegetation_2])[index]

def f(t, period, spectral_signature_1, spectral_signature_2):
    return np.sin((t%period) * np.pi/period )* spectral_signature_1 + np.sin((1-(t%period)) * np.pi/period ) * spectral_signature_2 + 0.5

grass_time_series = np.zeros((342, 10))
for i in range(342):
    grass_time_series[i] = f(i, 73, spring_vegetation, summer_vegetation) + np.random.normal(0, 0.0008, 10)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
slider_ax = plt.axes([0.1, 0.1, 0.65, 0.03])
slider = Slider(slider_ax, "Band", 0, 9, valinit=0, valstep=1)
ax.plot(range(0,342*5,5), grass_time_series[:, 0])

def update(val):
    ax.clear()
    ax.plot(range(0,342*5,5), grass_time_series[:, int(slider.val)])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

