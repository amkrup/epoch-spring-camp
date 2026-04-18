import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
from collections import Counter

data = np.array([
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
])

test_data = np.array([
    [118, 6.2],
    [160, 7.3],
    [185, 7.7],
])

X = data[:, :2].astype(float)
y = data[:, -1]

x_range = X[:, 0].max() - X[:, 0].min()
y_range = X[:, 1].max() - X[:, 1].min()

X_MIN = X[:, 0].min() - 0.4 * x_range
X_MAX = X[:, 0].max() + 0.4 * x_range
Y_MIN = X[:, 1].min() - 0.4 * y_range
Y_MAX = X[:, 1].max() + 0.4 * y_range

def euclideanDistance(p, q): # this normalizes the distances, we need this so that our decision boundaries are not vertical
    p, q = np.array(p), np.array(q)
    scale = np.array([X_MAX - X_MIN, Y_MAX - Y_MIN])
    return np.sqrt(np.sum(((p - q) / scale) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

    def predict_one(self, x):
        distances = [(euclideanDistance(self.X[i], x), self.y[i]) for i in range(len(self.X))]
        k_labels = [label for _, label in sorted(distances)[:self.k]]
        return Counter(k_labels).most_common(1)[0][0]


class wKNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

    def predict_one(self, X):
        distances = []
        for i in range(len(self.X)):
            distance = euclideanDistance(self.X[i], X)
            distances.append((distance, self.y[i]))

        k_nearest = sorted(distances)[:self.k]

        class_weights = {}
        for dist, label in k_nearest:
            weight = 1 / (dist + 1e-10)
            if label not in class_weights:
                class_weights[label] = 0
            class_weights[label] += weight

        return max(class_weights, key=class_weights.get)

def get_grid_predictions(train_X, train_y, k, w, resolution=80):
    xx, yy = np.meshgrid(
        np.linspace(X_MIN, X_MAX, resolution),
        np.linspace(Y_MIN, Y_MAX, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    label_to_int = {'Apple': 0, 'Banana': 1, 'Orange': 2}
    knn = wKNN(k=k) if w else KNN(k=k)
    knn.fit(train_X, train_y)
    Z = np.array([label_to_int[knn.predict_one(p)] for p in grid_points])

    return xx, yy, Z.reshape(xx.shape)


cmap = ListedColormap(['#8b1a1a', '#7a6600', '#7a3300']) # custom color map for Apple, Banana, Orange decision areas
color_map = {'Apple': '#ef4444', 'Banana': '#facc15', 'Orange': '#fb923c'}

fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor('#0e0e0e') # background
ax = fig.add_axes([0.10, 0.18, 0.85, 0.75]) # starts 10% from the left, starts 18% from the bottom,
                                            # takes up 85% of the figure width, takes up 75% of the figure height
ax_slider = fig.add_axes([0.35, 0.06, 0.55, 0.04])
slider = Slider(
    ax=ax_slider, label='k', valmin=1, valmax=7, # funny stuff happened when ax=ax!
    valinit=3, valstep=2, color='#444', track_color='#222'
)
slider.vline.set_visible(False) # removes the red mark which was at val = 3
ax_slider.set_facecolor('#1a1a1a')
slider.label.set_color('#aaa')
slider.valtext.set_color('#fff')

wknn_slider = fig.add_axes([0.10, 0.06, 0.10, 0.04])
slider_w = Slider(
    ax=wknn_slider, label='', valmin=0, valmax=1,
    valinit=0, valstep=1, color='#444', track_color='#222'
)
wknn_slider.text(-0.15, 0.5, 'KNN', transform=wknn_slider.transAxes,
          color='#aaa', va='center', ha='right')

wknn_slider.text(1.15, 0.5, 'wKNN', transform=wknn_slider.transAxes,
          color='#aaa', va='center', ha='left')
slider_w.vline.set_visible(False)
slider_w.valtext.set_visible(False)
slider_w.label.set_visible(False)
wknn_slider.set_facecolor('#1a1a1a')


def draw(k, w):
    ax.cla()

    ax.set_facecolor('#111111')
    ax.set_xlabel('weight (g)', color='#888', fontsize=10)
    ax.set_ylabel('length (cm)', color='#888', fontsize=10)
    ax.tick_params(colors='#666')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.grid(True, color='#2a2a2a', linewidth=0.5)
    mode = "wKNN" if w else "KNN"
    ax.set_title(f'{mode} Classifier — k = {k}  |  drag points to update',
                 color='#aaa', fontsize=10, pad=10)

    xx, yy, Z = get_grid_predictions(X, y, k, w)
    ax.pcolormesh(xx, yy, Z[:-1, :-1], cmap=cmap, shading='auto', alpha=0.45) # Z[:-1, :-1] becuase pcolormesh shades
                                                                                    # in between corners of quadrilaterals

    for label in ['Apple', 'Banana', 'Orange']:
        mask = y == label # mask is a boolean np array
        pts = X[mask] # this acts as a filter, and returns only the coordinates which are of a particular label
        ax.scatter(pts[:, 0], pts[:, 1], color=color_map[label],
                   s=80, zorder=10, edgecolors='white', linewidths=0.8)

    knn = wKNN(k=k) if w else KNN(k=k)
    knn.fit(X, y)
    preds = knn.predict(test_data)

    for i, (pt, pred) in enumerate(zip(test_data, preds)):
        ax.scatter(pt[0], pt[1], marker='*', s=260, zorder=20,
                   facecolors='none', edgecolors=color_map[pred], linewidths=2)

    train_patches = [mpatches.Patch(color=color_map[l], label=f'{l} (train)') for l in color_map]
    star = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='none',
                      markersize=12, label='Test point (star = predicted class)', linestyle='None')
    ax.legend(handles=train_patches + [star],
              facecolor='#1a1a1a', edgecolor='#333',
              labelcolor='#aaa', fontsize=8, loc='upper left')

    fig.canvas.draw_idle()

def on_changed(val):
    k = int(slider.val)
    w = int(slider_w.val)
    draw(k, w)

dragging_index = None

def find_nearest(event):
    if event.inaxes is not ax:
        return None
    xy_pixels = ax.transData.transform(X)
    click = np.array([event.x, event.y])
    dists = np.linalg.norm(xy_pixels - click, axis=1)
    nearest = np.argmin(dists)
    return nearest if dists[nearest] < 12 else None

def on_press(event):
    global dragging_index
    dragging_index = find_nearest(event)

def on_motion(event):
    global dragging_index
    if dragging_index is None or event.inaxes is not ax:
        return
    X[dragging_index] = [event.xdata, event.ydata]
    draw(int(slider.val), int(slider_w.val))

def on_release(event):
    global dragging_index
    dragging_index = None

slider.on_changed(on_changed)
slider_w.on_changed(on_changed)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

draw(3, int(slider_w.val))
plt.show()