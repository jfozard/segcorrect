
# Tool for segmenting bladders and other 3D objects from tiff stacks

from __future__ import print_function


import numpy as np
import numpy.linalg as la
import sys
import math
import itertools
import heapq
import os

import time

import scipy.ndimage as nd

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMessageBox

except ImportError:
    from PyQt4 import QtGui, QtCore
    from PyQt4 import QtGui as QtWidgets
    from PyQt4.QtGui import QMessageBox

    
    from PyQt4.QtCore import Qt
#from PyQt4.QtOpenGL import *
import numpy.random as npr

from functools import wraps

from orthoview_labels import OrthoView

from mini_controller import WorldController

import argparse

from log_tool import *

def menu(name, shortcut=None):
    def decorator(f):
        f.menu = name
        f.shortcut = shortcut
        return f
    return decorator

def toolbar(f):
    f.toolbar = True
    return f
    

def downsample_mask(ma, k):
    return ma.reshape(ma.shape[0]//k, k, ma.shape[1]//k, k, ma.shape[2]//k, k).sum(axis=(1,3,5))>(k*k*k/2)

def upsample_mask(ma, k):
    ##return np.kron(ma, np.ones((k,k,k), dtype=ma.dtype))
    if k>1:
        return nd.zoom(ma, [k,k,k], order=0)
    else:
        return ma


def downsample_max(ma, k):
    return ma.reshape(ma.shape[0]//k, k, ma.shape[1]//k, k, ma.shape[2]//k, k).max(axis=(1,3,5))


def downsample(ma, k):
    return ma.reshape(ma.shape[0]//k, k, ma.shape[1]//k, k, ma.shape[2]//k, k).sum(axis=(1,3,5))/k/k/k

def upsample(ma, k):
    return nd.zoom(ma, k, order=1)


class LSControl(QtWidgets.QWidget):
    def __init__(self, sw):
        QtWidgets.QWidget.__init__(self)
        self.sw = sw
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        self.params = sw.get_ls_params()
        self.wl = {}
        for i, p in enumerate(self.params):
            v = self.params[p]
            sb = QtWidgets.QDoubleSpinBox()
            sb.setMinimum(min(10*v, -1000.0))
            sb.setMaximum(max(10*v, 1000.0))
            sb.setDecimals(6)
            sb.setValue(v)
            layout.addWidget(QtWidgets.QLabel(p), i, 0)
            layout.addWidget(sb, i, 1)
            sb.valueChanged.connect(self.set_values)
            self.wl[p] = sb

    def set_values(self, val):
        self.sw.set_ls_params(dict((p, self.wl[p].value()) for p in self.params) )



def get_dirs(self):
    while True:
        r = QtWidgets.QFileDialog.getOpenFileName(self, 'Segmented Stack', '.', 'TIFF files (*.tif)')
        if type(r)==tuple:
            r = r[0]
        if r:
            label_filename = str(r)
            break


        
    while True:
        r = QtWidgets.QFileDialog.getOpenFileName(self, 'Wall Image Stack', os.path.dirname(label_filename), 'TIFF files (*.tif)')
        if type(r)==tuple:
            r = r[0]
        if r:
            signal_filename = str(r)
            break

    autosave_dir = os.path.dirname(label_filename)+'/'+'autosave_'+os.path.basename(label_filename[:-4])
    try:
        os.makedirs(autosave_dir)
    except OSError:
        pass
    return label_filename, signal_filename, autosave_dir
        
        

class SegmentWindow(QtWidgets.QMainWindow):
    def __init__(self, label_file, signal_file, signal_directory, replay, quit_after=False, spacing=None, log=None, autosave_dir=None):

        self.label_opacity = 0.2
        self.ct_weight = 0.0
        self.grey_labels = False

        self.cell_graph = None

        self.show_celltypes = True
        self.cell_prop = None

        self.threshold = 1
        self.autosave_dir = autosave_dir
        print('as', self.autosave_dir)

        QtWidgets.QMainWindow.__init__(self)

        self.status_bar = self.statusBar()
        
        if log is None:
            if label_file:
                base_name = os.path.splitext(os.path.basename(label_file))[0]
                log = log_name_td_filename(base_name)
            elif signal_file:
                base_name = os.path.splitext(os.path.basename(signal_file))[0]
                log = log_name_td_filename(base_name)
            else:
                log = log_td_filename()


        if not label_file and not signal_file:
            label_file, signal_file, signal_directory = get_dirs(self)
            
            
        log_file = open(log, 'w')
        if spacing:
            self.wc = WorldController(log_file, np.array(map(float, spacing.split(','))), autosave_dir=self.autosave_dir)
        else:
            self.wc = WorldController(log_file, None, autosave_dir=self.autosave_dir)


        if signal_file:
            self.wc.load_signal(signal_file, '%' in signal_file)
            
        if label_file:
            self.wc.load_label(label_file, '%' in label_file)
            if os.path.exists(label_file[:-3]+'csv'):
                self.wc.read_celltypes(label_file[:-3]+'csv')

        if replay:
            self.wc.replay_log(replay)
            if quit_after:
                quit()
        
        
        self.init_view()
        self.update_stack()

        
    def init_view(self):
        self.tbar = QtWidgets.QToolBar()
        self.addToolBar(self.tbar)
        self.t_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.t_slider.setMaximum(255)
        self.t_slider.setValue(int(self.label_opacity*255))
        self.tbar.addWidget(self.t_slider)

        self.addToolBarBreak()
        self.actionbar = QtWidgets.QToolBar()
        self.addToolBar(self.actionbar)
#        self.actionbar.addWidget(QtWidgets.QLabel("test"))
        
        cmap = self.wc.gen_colmap(self.cell_prop, self.show_celltypes, self.wc.get_selected(), ct_weight=self.ct_weight, grey_labels=self.grey_labels)

        self.widget = OrthoView(self.wc.get_label_stack(),  self.wc.get_signal_stack(), self.wc.get_spacing(), cmap, window=self)
        self.widget.set_label_opacity(self.label_opacity)

        
        self.wc.update_callbacks.append(self.world_updated)
        
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.setMaximum(255)
        self.z_slider.setValue(int((self.widget.zoom-0.1)/2.9*255))
        self.tbar.addWidget(self.z_slider)


        self.ct_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ct_slider.setMaximum(255)
        self.ct_slider.setValue(int(self.ct_weight*255))
        self.tbar.addWidget(self.ct_slider)

        
        self.setCentralWidget(self.widget) 
        self.z_slider.valueChanged.connect(self.z_changed)
        self.t_slider.valueChanged.connect(self.t_changed)
        self.ct_slider.valueChanged.connect(self.ct_changed)

        
        
        self.ctxmenu = QtWidgets.QMenu("ctx")
        ctx_submenus = {}

        for u in dir(self):
            if u.startswith("action_"):
                f = getattr(self, u)
                try:
                    m = f.menu
                    print(u, f.menu)
                    if m not in ctx_submenus:
                        ctx_submenus[m] = self.ctxmenu.addMenu(m)
                except:
                    pass

        for u in dir(self):
            if u.startswith("action_"):
                n = u[7:]
                action = QtWidgets.QAction(n, self)
                self.addAction(action)
                f = getattr(self, u)
                try:
                    m = f.menu
                    ctx_submenus[m].addAction(action)
                except:
                    print('not in submenus', f.menu)
                    self.ctxmenu.addAction(action)
                if f.shortcut:
                    action.setShortcut(f.shortcut)
                if hasattr(f, 'toolbar'):
                    self.actionbar.addAction(action)
                    
                action.triggered.connect(f)
#                self.parent().tbar.addAction(action)
        self.ctx_submenus = ctx_submenus


        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        save_action = QtWidgets.QAction('&Save as ...', self) 
        save_action.setShortcut('Ctrl+S')
        quit_action = QtWidgets.QAction('&Quit', self)
        quit_action.setShortcut('Ctrl+Q')                                        
        fileMenu.addAction(save_action)
        fileMenu.addAction(quit_action)
        save_action.triggered.connect(self.action_write_label_stack)
        quit_action.triggered.connect(self.quit_action)
        

        
        self.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)


    def quit_action(self):
        self.close()

    


    def closeEvent(self, event):

        reply = QMessageBox.question(
                              self,
                              'Quit',
                              'Are you sure you want to quit?',
                              QMessageBox.Yes | QMessageBox.No,
                              QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        
    def contextMenuEvent(self, ev):
        pos = self.mapToGlobal(ev.pos())
        self.ctxmenu.exec_(pos)


    
    @menu('segment')
    def action_subtract_bg(self):
        self.wc.subtract_bg()

    @menu('signal')
    def action_load_additional_signal(self):
        r = QtWidgets.QFileDialog.getOpenFileName(self, 'Stack', '.', 'TIFF files (*.tif)')
        if type(r)==tuple:
            r = r[0]
        if r:
            filename = r

            filename = str(filename)
            self.wc.load_signal(filename)

    @menu('signal')
    def action_load_additional_signal_rgb(self):
        r = QtWidgets.QFileDialog.getOpenFileName(self, 'Stack', '.', 'TIFF files (*.tif);;PNG files (*.png)')
        if type(r)==tuple:
            r = r[0]
        if r:
            filename = r
            filename = str(filename)
            self.wc.load_signal_rgb(filename)
        
    @menu('signal')
    def action_select_active_signal(self):
        all_img_stack_names = list(self.wc.get_signal_names())
        stack_name, flg = QtWidgets.QInputDialog.getItem(self, "Choose displayed signal", "name: ", all_img_stack_names)
        if flg:
            stack_name = str(stack_name)
            self.wc.select_active_signal(stack_name)


    @menu('view')
    def action_set_zoom(self):
        txt, flg = QtWidgets.QInputDialog.getText(self, "Zoom factor", "zoom:")
        if flg:
            txt = str(txt)
            try:
                self.widget.zoom = float(txt)
                self.widget.update_labels()
            except ValueError:
                pass
            
        
  
    @menu('signal')
    def action_apply_power(self):
        power, flg = QtWidgets.QInputDialog.getDouble(self, "Power",
                                                  "Power", 0.5, decimals=2)
        if flg:
            self.wc.apply_power(power)

    @menu('signal')
    def action_flip_z(self):
        self.wc.flip_z()

    @menu('segment', 'Ctrl+Z')
    def action_undo(self):
        self.wc.undo()


    @menu('segment')
    def action_write_label_stack(self):
        fn = QtWidgets.QFileDialog.getSaveFileName(self, 'Stack', '.', 'TIFF (*.tif)')
        if type(fn)==tuple:
            fn = fn[0]
        fn = str(fn)
        if fn[:-4]!='.tif':
            fn += '.tif'
        self.wc.write_label_tiff(fn)
        self.wc.write_celltypes(fn[:-3]+'csv')

        
    @menu('segment')
    def action_write_celltypes(self):
        fn = QtWidgets.QFileDialog.getSaveFileName(self, 'Celltypes', '.', 'CSV (*.csv)')
        if type(fn)==tuple:
            fn = fn[0]
        fn = str(fn)
        if fn:
            self.wc.write_celltypes(fn)

    @menu('segment')
    def action_read_celltypes(self):
        fn = QtWidgets.QFileDialog.getOpenFileName(self, 'Celltypes', '.', 'CSV (*.csv)')
        if type(fn)==tuple:
            fn = fn[0]
        fn = str(fn)
        if fn:
            self.wc.read_celltypes(fn)
        

    @menu('signal')
    def action_write_signal_stack(self):
        fn = QtWidgets.QFileDialog.getSaveFileName(self, 'Stack', '.', 'TIFF (*.tif)')
        if type(fn)==tuple:
            fn = fn[0]
        fn = str(fn)
        self.wc.write_signal_tiff(fn)

    @toolbar
    @menu('view') 
    def action_set_cell_prop(self):
        cell_props = ['label'] + list(self.wc.get_cell_props())

        prop_name, flg = QtWidgets.QInputDialog.getItem(self, "Choose displayed cell property", "prop: ", cell_props)

        if not flg:
            return
        
        prop_name = str(prop_name)
 
        if prop_name == 'label':
            self.cell_prop = None
        else:
            self.cell_prop = prop_name

        self.update_colmap()
        self.widget.update_labels()

    @menu('view')
    def action_show_mean_signal(self):
        self.wc.calc_mean_signal()
        self.cell_prop = 'mean_signal'

    @menu('view') # ????
    def action_show_mean_interior_signal(self):
        self.wc.calc_mean_interior_signal()
        self.cell_prop = 'mean_interior_signal'


    @menu('segment')
    def action_set_omitted(self):
        txt, flg = QtWidgets.QInputDialog.getText(self, "Omitted", "celltypes")
        omitted = map(int, str(txt).split())
        self.wc.set_omitted(omitted)

    @menu('view')
    def action_toggle_grey_labels(self):
        self.grey_labels = not self.grey_labels
        self.update_colmap()
        self.widget.update_labels()
    

    @menu('view')
    def action_set_label_opacity(self):
        txt, flg = QtWidgets.QInputDialog.getText(self, "Label opacity", "opacity")
        try:
            alpha = float(txt)
            self.label_opacity = alpha
            self.update_colmap()
            self.widget.update_labels()
        except ValueError:
            pass

    @menu('view')
    def action_set_ct_weight(self):
        txt, flg = QtWidgets.QInputDialog.getText(self, "Celltype weight", "0-1")
        try:
            alpha = float(txt)
            self.ct_weight = alpha
            self.update_colmap()
            self.widget.update_labels()
        except ValueError:
            pass


    @menu('segment')
    def action_select_by_celltype(self):
        txt, flg = QtWidgets.QInputDialog.getText(self, "Selected", "celltypes")




    @menu('segment')
    def action_select_large(self):
        small, flg = QtWidgets.QInputDialog.getDouble(self, "Lower limit",
                                                  "Lower", 10.0, decimals=4)

        self.wc.select_large(small)

    @menu('segment')
    def action_select_small(self):
        large, flg = QtWidgets.QInputDialog.getDouble(self, "Upper limit",
                                                  "Upper", 0.1, decimals=4)
        self.wc.select_small(large)


    def t_changed(self, value):
        self.label_opacity = value/255.0
        self.update_colmap()
        self.widget.update_labels()

    def ct_changed(self, value):
        self.ct_weight = value/255.0
        self.update_colmap()
        self.widget.update_labels()

        
    def z_changed(self, value):
        self.widget.zoom = 0.1+2.9*(value/255.0)
        self.widget.update_labels()

        
    @menu('segment')
    def action_seed_minima(self):
        r, flg = QtWidgets.QInputDialog.getDouble(self, "Radius",
                                                  "r", 3.0, decimals=4)
        if not flg:
            return

        self.wc.seed_minima(r)

    @menu('segment', QtCore.Qt.Key_W)
    def action_watershed_from_labels(self):
        self.wc.watershed_from_labels()

    @menu('segment')
    def action_watershed_no_labels(self):
        h, flg = QtWidgets.QInputDialog.getDouble(self, "h_minima level",
                                                  "h", 1.0, decimals=1)
        if not flg:
            return

        self.wc.watershed_no_labels(h=h)


    @toolbar
    @menu('segment', QtCore.Qt.Key_S)
    def action_add_seed(self):
        v = self.widget.slice
        self.wc.add_seed(v)

    @menu('segment', QtCore.Qt.Key_A)
    def action_add_seed_selected(self):
        v = self.widget.slice
        self.wc.add_seed(v, use_selected=True)
        

    def drag_callback(self):
        v = self.wc.get_label_point(self.widget.slice)
        self.wc.include_selected(v)
        
    @toolbar
    @menu('segment', QtCore.Qt.Key_1)
    def action_select(self):
        v = self.wc.get_label_point(self.widget.slice)
        self.wc.set_selected([v])

    @toolbar
    @menu('segment', QtCore.Qt.Key_2)
    def action_select_add(self):
        v = self.wc.get_label_point(self.widget.slice)
        self.wc.add_selected(v)

    @toolbar
    @menu('segment', QtCore.Qt.Key_3)
    def action_select_merge(self):
        self.wc.merge_selected()


    @toolbar
    @menu('segment', QtCore.Qt.Key_Delete)
    def action_delete(self):
        self.wc.delete_selected()

    @toolbar
    @menu('segment',  QtCore.Qt.Key_C)
    def action_set_celltype(self):
        ct, flg = QtWidgets.QInputDialog.getInt(self, "Celltype",
                                                  "Celltype", 1)
        if not flg:
            return
        self.wc.set_celltype(ct)

        
    @menu('segment', QtCore.Qt.Key_8)
    def action_set_celltype0(self):
        self.wc.set_celltype(0)

    @menu('segment', QtCore.Qt.Key_9)
    def action_set_celltype1(self):
        self.wc.set_celltype(1)



    @menu('signal')
    def action_copy_signal_stack(self, value):
        self.wc.copy_signal_stack()
        

    @menu('signal')
    def action_invert_signal(self, value):
        self.wc.invert_signal()
        

        
    @menu('signal')
    def action_blur_stack(self, value):
        radius, flg = QtWidgets.QInputDialog.getDouble(self, "Radius",
                                                  "Radius", 3.0, decimals=2)
        if flg:
            self.wc.blur_stack(radius)

    @menu('segment')
    def action_dilate_labels(self):
        self.wc.dilate_labels()

    @toolbar
    @menu('segment', QtCore.Qt.Key_R)
    def action_resegment(self):
        self.wc.resegment()

    @toolbar
    @menu('segment', QtCore.Qt.Key_P)
    def action_split_plane(self):
        self.wc.split_plane()

        

    @menu('segment') 
    def action_select_by_prop(self, value):
        props = self.wc.get_cell_props()        
        cond, flg = QtWidgets.QInputDialog.getText(self, "Condition", "statement")
        cond = str(cond)
        self.wc.select_by_prop(cond)

    @menu('signal')
    def action_equalize_stack(self, value):
        self.wc.equalize_stack()



    def world_updated(self, *msg):
        print('world updated', msg)
        self.update_colmap()
        self.update_stack()

    def update_stack(self):
        self.widget.update(self.wc.get_label_stack(), self.wc.get_signal_stack())
#        self.gl_widget.widget.update_stack()

    def update_colmap(self):
        cmap_w = self.wc.gen_colmap(self.cell_prop, self.show_celltypes, self.wc.get_selected(), ct_weight=self.ct_weight, grey_labels=self.grey_labels)
        self.widget.set_label_opacity(self.label_opacity)
        self.widget.update_colmap(cmap_w)
        selected=self.wc.get_selected()
        celltypes = self.wc.get_celltypes()
        self.status_bar.showMessage(repr((selected,[celltypes.get(i, None) for i in selected])))


        
def main():

    parser = argparse.ArgumentParser(description='Generate surface mesh from stack')

 #   group = parser.add_mutually_exclusive_group(required=True)


 
    parser.add_argument('--image_dir', action='store_true', help='read signal as directory')

    parser.add_argument('--spacing', metavar='spacing', type=str, default='', help='override image spacing')

    parser.add_argument('-r', '--replay', metavar='replay', default=None, help='replay log file')
    parser.add_argument('-l', '--labels', metavar='labels', default=None, help='labelled stack')
    parser.add_argument('-i', '--image', metavar='signal', default=None, help='signal stack')
    parser.add_argument('--log', metavar='log', default=None, help='log filename')

    parser.add_argument('-q', '--quit_after', action='store_true', help='quit after replay')

    parser.add_argument('--autosave_dir', metavar='autosave_dir', default=None, help='directory for automatic saves')

    
    args = parser.parse_args()
    print(args)

    app = QtWidgets.QApplication(['Stack segmentation'])


    window = SegmentWindow(args.labels, args.image, args.image_dir, args.replay, args.quit_after, args.spacing, args.log, args.autosave_dir)

#    window.setFixedSize(1000,1000)
    window.show()
    app.exec_()

main()        
        



