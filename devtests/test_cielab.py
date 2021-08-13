import ltempy as wt
from ltempy import np, plt

wt.cielab_cmap()
X = np.linspace(0,1,100)
x, y = np.meshgrid(X, X)
data = x + 1j * y

plt.imshow(wt.cielab_image(data))
plt.title("intensity, uniform")
plt.show()

plt.imshow(wt.cielab_image(data, brightness='amplitude'))
plt.title("amplitude, uniform")
plt.show()

plt.imshow(wt.cielab_image(data, alpha='intensity'))
plt.title("intensity, intensity")
plt.show()

plt.imshow(wt.cielab_image(data, alpha='amplitude'))
plt.title("intensity, amplitude")
plt.show()

plt.imshow(wt.rgba(data), cmap='viridis')
plt.title("rgba, cmap=viridis")
plt.show()
