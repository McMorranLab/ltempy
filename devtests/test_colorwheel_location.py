# %%
import matplotlib.pyplot as plt
import ltempy as lp
import numpy as np

X = np.linspace(-1, 1, 128) * 3
Y = np.linspace(-1, 1, 64) * 3
x, y = np.meshgrid(X, Y)

f = np.exp(1j*np.sin(x)*y**2) * np.sin(x)

for origin in ["upper", "lower"]:
    for align_x in ["left", "right"]:
        for align_y in ["top", "bottom"]:
            fig, [[ax]] = lp.subplots()
            ax.set_title(f"Align_x: {align_x}\nAlign_y: {align_y}\nOrigin: {origin}")
            ax.origin = origin
            ax.cielab(f)
            ax.quiver(f, step=3)
            ax.colorwheel(brightness='amplitude', align_x=align_x, align_y=align_y)
            plt.show()