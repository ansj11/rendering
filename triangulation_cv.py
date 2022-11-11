import cv2
import numpy as np

from IPython import embed

path = "/Users/anshijie/debug/mesh/0.png"

mask = cv2.imread(path, 0)

rect = cv2.boundingRect(mask)

ys, xs = np.where(mask > 0)


subdiv = cv2.Subdiv2D(rect)


for i, (y, x) in enumerate(zip(ys, xs)):
    try:
        subdiv.insert((float(x), float(y)))
    except:
        embed()


triangles = subdiv.getTriangleList()

print(triangles)

embed()
