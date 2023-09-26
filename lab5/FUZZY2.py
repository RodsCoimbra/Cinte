from simpful import *
import matplotlib.pylab as plt
from numpy import linspace, array

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['poor', 'average',
                   'good'], universe_of_discourse=[0, 10])
FS.add_linguistic_variable("service", TLV)
FS.add_linguistic_variable("quality", TLV)

O1 = TriangleFuzzySet(0, 0, 13,   term="low")
O2 = TriangleFuzzySet(0, 13, 25,  term="medium")
O3 = TriangleFuzzySet(13, 25, 25, term="high")
FS.add_linguistic_variable("tip", LinguisticVariable(
    [O1, O2, O3], universe_of_discourse=[0, 25]))

FS.add_rules([
    "IF (quality IS poor) OR (service IS poor) THEN (tip IS low)",
    "IF (service IS average) THEN (tip IS medium)",
    "IF (quality IS good) OR (service IS good) THEN (tip IS high)"
])

FS.set_variable("quality", 2)
FS.set_variable("service", 4)
print(FS.inference(['tip']))
#FS.produce_figure()



# Plotting surface
xs = []
ys = []
zs = []
DIVs = 20
for x in linspace(0,10,DIVs):
    for y in linspace(0,10,DIVs):
        FS.set_variable("quality", x)
        FS.set_variable("service", y) 
        tip = FS.inference()['tip']
        xs.append(x)
        ys.append(y)
        zs.append(tip)
xs = array(xs)
ys = array(ys)
zs = array(zs)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx, yy = plt.meshgrid(xs,ys)

ax.plot_trisurf(xs,ys,zs, vmin=0, vmax=25, cmap='gnuplot2')
ax.set_xlabel("Quality")
ax.set_ylabel("Service")
ax.set_zlabel("Tip")
ax.set_title("Simpful", pad=20)
ax.set_zlim(0, 25)
plt.tight_layout()
plt.show()
