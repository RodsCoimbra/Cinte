from mpl_toolkits.mplot3d import Axes3D
from simpful import *
import matplotlib.pylab as plt
from numpy import linspace, array


FS = FuzzySystem()

TEMP1 = TriangleFuzzySet(0, 0, 50,   term="cold")
TEMP2 = TriangleFuzzySet(30, 50, 70, term="warm")
TEMP3 = TriangleFuzzySet(50, 100, 100, term="hot")

FS.add_linguistic_variable("Temperature", LinguisticVariable(
    [TEMP1, TEMP2, TEMP3], universe_of_discourse=[0, 100]))
Clock1 = TriangleFuzzySet(0, 0, 1.5, term="low")
Clock2 = TriangleFuzzySet(0.5, 2, 3.5, term="medium")
Clock3 = TriangleFuzzySet(2.5, 4, 4, term="high")
FS.add_linguistic_variable("Clock", LinguisticVariable(
    [Clock1, Clock2, Clock3], universe_of_discourse=[0, 4]))
Fan1 = TriangleFuzzySet(0, 0, 3500, "slow")
Fan2 = TriangleFuzzySet(2000, 3000, 4000, "medium")
Fan3 = TriangleFuzzySet(2500, 6000, 6000, "fast")
FS.add_linguistic_variable("Fan", LinguisticVariable(
    [Fan1, Fan2, Fan3], universe_of_discourse=[0, 6000]))

FS.add_rules([
    "IF (Temperature IS cold) AND (Clock IS low) THEN (Fan IS slow)",
    "IF (Temperature IS cold) AND (Clock IS medium) THEN (Fan IS medium)",
    "IF (Temperature IS cold) AND (Clock IS high) THEN (Fan IS medium)",
    "IF (Temperature IS warm) AND (Clock IS low) THEN (Fan IS slow)",
    "IF (Temperature IS warm) AND (Clock IS medium) THEN (Fan IS medium)",
    "IF (Temperature IS warm) AND (Clock IS high) THEN (Fan IS fast)",
    "IF (Temperature IS hot) AND (Clock IS high) THEN (Fan IS fast)",
    "IF (Temperature IS hot) AND (Clock IS low) THEN (Fan IS medium)",
    "IF (Temperature IS hot) AND (Clock IS medium) THEN (Fan IS fast)"
])

FS.set_variable("Temperature", 100)
FS.set_variable("Clock", 4)
print(FS.inference(['Fan']))

# Plotting surface
xs = []
ys = []
zs = []
DIVs = 20
for x in linspace(0, 100, DIVs):
    for y in linspace(0, 4, DIVs):
        FS.set_variable("Temperature", x)
        FS.set_variable("Clock", y)
        speed = FS.inference()['Fan']
        xs.append(x)
        ys.append(y)
        zs.append(speed)
xs = array(xs)
ys = array(ys)
zs = array(zs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx, yy = plt.meshgrid(xs, ys)

ax.plot_trisurf(xs, ys, zs, vmin=0, vmax=6000, cmap='gnuplot2')
ax.set_xlabel("Temperature")
ax.set_ylabel("Clock")
ax.set_zlabel("Fan")
ax.set_title(
    "RELATIONSHIP BETWEEN CPU FAN SPEED, TEMPERATURE AND CLOCK", pad=20)
ax.set_zlim(0, 6000)
plt.tight_layout()
plt.show()

FS.produce_figure()
