from typing import Dict, Any

import matplotlib
matplotlib.use('module://kivymatplot.backend_kivy')
from matplotlib.figure import Figure
from numpy import arange, sin, pi
from kivy.app import App
from kivy.core.window import Window
import numpy as np
from matplotlib.mlab import griddata

from kivymatplot.backend_kivyagg import FigureCanvas, NavigationToolbar2Kivy

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from matplotlib.transforms import Bbox
from kivy.uix.button import Button
from kivy.graphics import Color, Line, Rectangle

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class KivyMplKArmedViewer(App):
    title = 'K-Armed Bandit Experiment Viewer'
    
    def build(self):
        fl = BoxLayout(orientation="vertical")
        a = Button(
            text="Close", height=150, size_hint_y=None)
        a.bind(on_press=self._callback)
        fl.add_widget(self.fig.canvas)
        fl.add_widget(a)
        return fl

    def _callback(self, _):
        # Clear the existing figure and re-use it
        self.fig.clf()
        self.prepare_plot()
        self.fig.canvas.draw_idle()

    def prepare_plot(self):
        X = np.arange(-508, 510, 203.2)
        Y = np.arange(-508, 510, 203.2)
        X, Y = np.meshgrid(X, Y)
        Z = np.random.rand(6, 6)
        contour = plt.contourf(
            X, Y, Z, 100, zdir='z', offset=1.0, 
            cmap=cm.hot)
        plt.colorbar(contour)
        self.ax.set_ylabel('Y [mm]')
        self.ax.set_title('NAILS surface')
        self.ax.set_xlabel('X [mm]')

    def open(self):
        matplotlib.rcParams.update({
            'font.size': min(Window.width, Window.height)/80})
        self.fig, self.ax = plt.subplots()
        self.prepare_plot()
        self.run()

    def close(self):
        self.stop()
        
    def update(self, state: Dict[str, Any]): ...


if __name__ == '__main__':
    KivyMplKArmedViewer().open()
    