import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from util.util import make_tensor_1d

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


######################################################################


def plot_poly_elem(ax, elem, i_elem, facecolor='0.4', alpha=0.3, edgecolor='black', label='', is_closed=False,
                   linewidth=1):
    assert elem.ndim == 2
    x = elem[:, 0].detach().cpu()
    y = elem[:, 1].detach().cpu()
    if i_elem > 0:
        label = None
    if is_closed:
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)
    else:
        ax.plot(x, y, alpha=alpha, color=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_lanes(ax, left_lanes, right_lanes, i_elem, facecolor='0.4', alpha=0.3, edgecolor='black', label='', linewidth=1):
    # assert len(left_lanes) == len(right_lanes)
    assert left_lanes.ndim == right_lanes.ndim == 2
    if i_elem > 0:
        label = None
    x_left = make_tensor_1d(left_lanes[:, 0])
    y_left = make_tensor_1d(left_lanes[:, 1])
    x_right = make_tensor_1d(right_lanes[:, 0])
    y_right = make_tensor_1d(right_lanes[:, 1])
    x = torch.cat((x_left, torch.flip(x_right, [0]))).detach().cpu()
    y = torch.cat((y_left, torch.flip(y_right, [0]))).detach().cpu()
    ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)

##############################################################################################

def plot_rectangles(ax, centroids, extents, yaws, label, facecolor, alpha=0.7, edgecolor='black'):
    n_elems = len(centroids)
    first_plt = True
    for i in range(n_elems):
        if first_plt:
            first_plt = False
        else:
            label = None
        height = extents[i][0]
        width = extents[i][1]
        angle = yaws[i]
        angle_deg = float(np.degrees(angle))
        xy = centroids[i] \
             - 0.5 * height * np.array([np.cos(angle), np.sin(angle)]) \
             - 0.5 * width * np.array([-np.sin(angle), np.cos(angle)])
        rect = Rectangle(xy, height, width, angle_deg, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, linewidth=1, label=label)

        ax.add_patch(rect)


##############################################################################################


def visualize_scene_feat(agents_feat_s, map_points_s, map_elems_availability_s, map_n_points_orig_s, opt):
    polygon_types = opt.polygon_types
    closed_polygon_types =  opt.closed_polygon_types

    centroids = [af['centroid'] for af in agents_feat_s]
    yaws = [af['yaw'] for af in agents_feat_s]
    print('agents centroids: ', centroids)
    print('agents yaws: ', yaws)
    print('agents speed: ', [af['speed'] for af in agents_feat_s])
    print('agents types: ', [af['agent_label_id'] for af in agents_feat_s])
    X = [p[0] for p in centroids]
    Y = [p[1] for p in centroids]
    U = [af['speed'] * np.cos(af['yaw']) for af in agents_feat_s]
    V = [af['speed'] * np.sin(af['yaw']) for af in agents_feat_s]
    fig, ax = plt.subplots()

    plot_props = {'lanes_mid': ('lime', 0.4), 'lanes_left': ('black', 0.3), 'lanes_right': ('black', 0.3),
                  'crosswalks': ('orange', 0.4)}
    pd = {}
    for i_type, poly_type in enumerate(polygon_types):
        pd[poly_type] = {}

        pd[poly_type]['elems_valid'] = map_elems_availability_s[i_type]
        pd[poly_type]['n_points_per_elem'] = map_n_points_orig_s[i_type]
        pd[poly_type]['elems_points'] = map_points_s[i_type]

        plot_poly_elems(ax, pd[poly_type],
                        facecolor=plot_props[poly_type][0], alpha=plot_props[poly_type][1],
                        edgecolor=plot_props[poly_type][0], label=poly_type,
                        is_closed=poly_type in closed_polygon_types, linewidth=1)

    plot_lanes(ax, pd['lanes_left'], pd['lanes_right'], facecolor='grey', alpha=0.3, edgecolor='black', label='Lanes')

    extents = [af['extent'] for af in agents_feat_s]
    plot_rectangles(ax, centroids[1:], extents[1:], yaws[1:])
    plot_rectangles(ax, [centroids[0]], [extents[0]], [yaws[0]], label='ego', facecolor='red', edgecolor='red')

    ax.quiver(X[1:], Y[1:], U[1:], V[1:], units='xy', color='b', label='Non-ego', width=0.5)
    ax.quiver(X[0], Y[0], U[0], V[0], units='xy', color='r', label='Ego', width=0.5)

    ax.grid()
    plt.legend()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image
