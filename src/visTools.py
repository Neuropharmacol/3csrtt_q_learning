import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from typing import Union, Optional, List, Dict, Callable, Any, Tuple
from types import ModuleType

class BasicPlot():
    def __init__(self, xlim=None, ylim=None, xlabel="", ylabel="",title="",
            save_path=None, figsize=(5,3), dpi=150,show=True):
        self.fig = plt.figure(figsize=figsize,dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(xlim) if xlim else None
        self.ax.set_ylim(ylim) if ylim else None
        self.save_path = save_path
        self.title = title
        self.show = show

    def __enter__(self):
        return(self)

    def __exit__(self,exc_type, exc_value, exc_traceback):
        self.option()
        plt.title(self.title)
        plt.tight_layout()
        if self.save_path:
            plt.savefig(self.save_path)
        if self.show:
            plt.show()

    def option(self):
        """This method is for additional graphic setting. 
        See DatePlot for example."""
        pass



class MultiPlot():
    """
    Usage:
        >>> with MultiPlot(grid=(1,2)) as p:
        >>>     ax = p.set_ax(0) 
    """
    def __init__(self, figsize=(5,3), dpi=150,grid=(2,2) ,suptitle="",
            save_path=None,show=True):
        self.fig = plt.figure(figsize=figsize,dpi=dpi)
        self.grid = grid
        self.save_path = save_path
        self.show = show

        plt.suptitle(suptitle)

    def set_ax(self,index,xlim=None, ylim=None, xlabel="", ylabel="",title=""):
        ax = self.fig.add_subplot(*self.grid,index)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim) if xlim else None
        ax.set_ylim(ylim) if ylim else None
        ax.set_title(title)
        return(ax)

    def __enter__(self):
        return(self)

    def __exit__(self,exc_type, exc_value, exc_traceback):
        self.option()
        #plt.title(self.title)
        plt.tight_layout()
        if self.save_path:
            plt.savefig(self.save_path)
        if self.show:
            plt.show()

    def option(self):
        """This method is for additional graphic setting. 
        See DatePlot for example."""
        pass

def create_patch_for_label(
    label_names: List[str],
    label_title: str = "",
    cmap_name: str = "tab10",
    color : Union[List[str], List[Tuple]] = None,
    line : bool = False,
    marker : Optional[List[str]] = None,
    markersize : Optional[int] = None,
    ) -> List[mpatches.Patch]:
    """Create list of patches for legend.

    Args:
        label_names : list of label names.
        label_title : title of label handle.
        cmap_name : colormap name.
        color : If color is specified, use this color set to display.
        line : legend becomes line style.
        marker : marker for Line2D.
        markersize : markersize for Line2D

    Examples:
        >>> patches = visTools.create_patch_for_label(label_names = ["test1", "test2", "test3"], color=["red","blue", "orange"] , line=True)
        >>> fig = plt.figure(figsize=(6,6), dpi=300 )
        >>> ax = fig.add_subplot(111)
        >>> ax.axes.xaxis.set_visible(False)
        >>> ax.axes.yaxis.set_visible(False)
        >>> plt.legend(handles=patches, frameon=False)
        >>> plt.show()
    """
    cmap = plt.get_cmap(cmap_name)
    patches = []
    #for c, name in zip(["blue","orange","green"],["男","女","不明"]):
    if marker is None:
        marker = [ None for i in range(len(label_names))]

    for i, name in enumerate(label_names):
        if color == None:
            c = cmap(i)
        else:
            c = color[i]
        if line:
            patch = Line2D([0], [0], color=c, label=name,
                        marker=marker[i], markersize=markersize)
        else:
            patch = mpatches.Patch(color=c, label=name)
        patches.append(patch)
    return(patches)

