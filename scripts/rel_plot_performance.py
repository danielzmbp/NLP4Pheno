import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


labels_flat = snakemake.params.labels

# Plotting


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


d = pd.read_csv(snakemake.input[0], sep="\t")
# select splits with highest f1 score
d_best = d[d["metric"] == "f1 score"].sort_values(
    "score", ascending=False).drop_duplicates("relation")[["relation", "split"]]
d_best = d_best.merge(d, on=["relation", "split"], how="left")
d_best.to_csv(snakemake.output[1], sep="\t", index=False)
m = d_best.pivot(columns="relation", values="score", index="metric")

data_mine = [m.columns.unique().to_list(), ("best split", m.values.tolist())]
N = len(labels_flat)
theta = radar_factory(N, frame='polygon')

data = data_mine
spoke_labels = data.pop(0)

fig, axs = plt.subplots(figsize=(6, 6), ncols=1,
                        subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

colors = ['b', 'r', 'g', 'm']

for title, case_data in data:
    axs.set_rgrids([0.2, 0.4, 0.6, 0.8])
    axs.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                  horizontalalignment='center', verticalalignment='center')
    for d, color in zip(case_data, colors):
        axs.plot(theta, d, color=color)
        axs.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    axs.set_varlabels(spoke_labels)
    axs.set_ylim(0, 1)


# add legend relative to top-left plot
labs = tuple(m.index.drop_duplicates().str.replace(
    "eval_", "", regex=True).to_list())
legend = axs.legend(labs, loc=(1, .95),
                    labelspacing=0.1, fontsize='small')

fig.text(1, 0, 'REL performance',
         horizontalalignment='center', color='black', weight='bold',
         size='large')


plt.savefig(snakemake.output[0], bbox_inches='tight', dpi=300)
