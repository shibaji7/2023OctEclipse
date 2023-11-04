#!/usr/bin/env python

"""
    rtiUtils.py: module to plot RTI plots with various y-axis transformation
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
import matplotlib.pyplot as plt

plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]
import datetime as dt

import matplotlib.dates as mdates
import numpy as np
from pysolar.solar import get_altitude_fast


def get_gridded_parameters(
    q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=True
):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[[xparam, yparam, zparam]]
    if rounding:
        plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    plotParamDF = plotParamDF.groupby([xparam, yparam]).mean().reset_index()
    plotParamDF = plotParamDF[[xparam, yparam, zparam]].pivot(xparam, yparam)
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y = np.meshgrid(x, y)
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
        np.isnan(plotParamDF[zparam].values), plotParamDF[zparam].values
    )
    return X, Y, Z

def get_zenith_angle(gdlat, glong, d):
    d = d.replace(tzinfo=dt.timezone.utc)
    za = 90.0 - get_altitude_fast(gdlat, glong, d)
    return za

class RTI(object):
    """
    Create plots for velocity, width, power, elevation angle, etc.
    """

    def __init__(
        self,
        drange,
        nGates=80,
        fig_title=None,
        num_subplots=1,
        ylim=[180, 3000],
    ):
        self.drange = drange
        self.nGates = 80
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(6, 3 * num_subplots), dpi=240)
        if fig_title:
            plt.suptitle(
                fig_title, x=0.075, y=0.99, ha="left", fontweight="bold", fontsize=15
            )
        self.ylim = ylim
        return

    def addParamPlot(
        self,
        radar,
        beam,
        title,
        vlim=[0, 30],
        xlabel="",
        zparam="p_l",
        label="Power (dB)",
        yscale="srange",
        cmap="jet",
        cbar=True,
        alpha=1,
        overlay_za=False,
    ):
        df = radar.df.copy()
        df = df[df.bmnum == beam]
        ax = self._add_axis()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H^{%M}"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size": 12, "fontweight": "bold"})
        ax.set_xlim(
            [mdates.date2num(self.drange[0]), mdates.date2num(self.drange[1])]
        )
        ax.set_ylabel("Slant Range (km)", fontdict={"size": 12, "fontweight": "bold"})
        X, Y, Z = get_gridded_parameters(
            df, xparam="time", yparam=yscale, zparam=zparam, rounding=False
        )
        im = ax.pcolormesh(
            X,
            Y,
            Z.T,
            lw=0.01,
            edgecolors="None",
            cmap=cmap,
            snap=True,
            vmax=vlim[1],
            vmin=vlim[0],
            shading="auto",
            alpha=alpha,
        )
        if cbar:
            self._add_colorbar(self.fig, ax, im, label=label)
        if title:
            ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        ax.set_ylim(self.ylim)
        if overlay_za:
            self.overlay_sza(
                radar.fov, ax, beam, [0, self.nGates],
                df.rsep.iloc[0], df.frang.iloc[0],
                yscale
            )
        return ax

    def overlay_sza(self, fov, ax, beam, gate_range, rsep, frang, yscale):
        """
        Add terminator to the radar
        """
        times = [
            self.drange[0] + dt.timedelta(minutes=i)
            for i in range(int((self.drange[1] - self.drange[0]).total_seconds() / 60))
        ]
        R = 6378.1
        gates = np.arange(gate_range[0], gate_range[1])
        dn_grid = np.zeros((len(times), len(gates)))
        for i, d in enumerate(times):
            d = d.replace(tzinfo=dt.timezone.utc)
            for j, g in enumerate(gates):
                gdlat, glong = fov[0][g, beam], fov[1][g, beam]
                angle = 90.0 - get_altitude_fast(gdlat, glong, d)
                dn_grid[i, j] = angle
        terminator = np.zeros_like(dn_grid)
        terminator[dn_grid > 90] = 1.0
        terminator[dn_grid <= 90] = 0.0
        gates = frang + (rsep * gates)
        times, gates = np.meshgrid(times, gates)
        ax.pcolormesh(
            times.T,
            gates.T,
            terminator,
            lw=0.01,
            edgecolors="None",
            cmap="gray_r",
            vmax=2,
            vmin=0,
            shading="nearest",
            alpha=0.3,
        )
        return

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax

    def _add_colorbar(
        self,
        fig,
        ax,
        im,
        label="",
        xOff=0,
        yOff=0,
    ):
        """
        Add a colorbar to the right of an axis.
        """
        cpos = [1.04 + xOff, 0.1 + yOff, 0.025, 0.8]
        cax = ax.inset_axes(cpos, transform=ax.transAxes)
        cb = fig.colorbar(im, ax=ax, cax=cax)
        cb.set_label(label)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return
