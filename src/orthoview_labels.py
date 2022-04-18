

import numpy as np
import numpy.random as npr
import scipy.ndimage as nd

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import Qt
except ImportError:
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
    from PyQt4.QtCore import Qt

from math import sqrt

gamma = 1.0

col1 = np.array([ 100, 255, 100 ], dtype=float)
col2 = np.array([ 255, 180, 255 ], dtype=float)

lut = np.zeros((512,3), dtype = np.uint8)
for i in range(256):
    j = i
    lut[i,:] = col1*(0.75*pow(float(j)/255.0, gamma) + 0.25)
    lut[i+256,:] = col2*(0.5*pow(float(j)/255, gamma) + 0.5)


binary_lut =  np.array([[50, 120, 50], [120, 50, 120], [180, 50, 50], [50, 50, 180]])


"""
orig_labels_lut = np.zeros((65536, 3), dtype=np.uint8)
for i in range(65536):
    j = i
    r = npr.rand(3)*255*(i>0)
    orig_labels_lut[i,:] = 0.5*r + 0.5
#    labels_lut[i+32768,:] = np.clip(255*np.array((0.9,0,0)) + np.array((0.1, 0.3, 0.3))*r, 0, 255)
"""

def to_qimg_gs(m, u, x_line=None, y_line=None):
    m2 = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    m2[:, :, :] = m[:, :, np.newaxis]
    return to_qimg_rgb(m2, x_line, y_line)


#@profile
def to_qimg_lut(m, u, x_line=None, y_line=None, labels_lut=None, pts=[], label_opacity=1.0):
    m2 = labels_lut[m & 65535,:]
    m2 = (col2[np.newaxis, np.newaxis, :]/2.0+m2/2.0)
    m4 = 255*u[:,:,np.newaxis]/max(1e-6,np.max(u))
    #print(m2.shape, m4.shape)
    m2 = label_opacity*m2+(1.0-label_opacity)*m4
    m2 = m2.astype(np.uint8)
    return to_qimg_rgb(m2, x_line, y_line, pts)


def to_qimg_float(m, u, x_line=None, y_line=None, pts=[]):

    m = pow((m)/np.max(m), gamma)
    m2 = col1[np.newaxis, np.newaxis, :].astype(float)*(0.75*m+0.25)[:,:,np.newaxis].astype(np.uint8)
                  

    return to_qimg_rgb(m2, x_line, y_line, pts)



def to_qimg_rgb(m, x_line=None, y_line=None, pts=[]):
    

    if pts:
        print(pts)
        xx, yy = np.mgrid[:height, :width]
        for p,r,c in pts:

            idx = (((xx-p[0])**2 + (yy-p[1])**2)<r**2).nonzero()
            m[idx[0], idx[1], : ] = np.array(c)[np.newaxis, np.newaxis, :]

    if x_line is not None:
        m[x_line, :, 0] = 255
    if y_line is not None:
        m[:, y_line, 0] = 255
    
    m = np.ascontiguousarray(m)
    height, width, channels = m.shape
    bytesPerLine = width*3

    return QtGui.QImage(QtCore.QByteArray(m.tostring()), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)    


class OrthoView(QtWidgets.QWidget):
    def __init__(self, stack, img_stack, spacing, colmap=binary_lut, window=None):
        QtWidgets.QWidget.__init__(self)
        self.stack = stack

        self.window = window
        
        if img_stack is not None:
            self.img_stack = img_stack
        else:
            self.img_stack = np.zeros(self.stack.shape)


        print('stack shape', self.stack.shape)
        print('img stack shape', self.img_stack.shape)

        self.label_opacity=0.5

        self.spacing = spacing
        self.colmap = colmap

        self.slice = [ stack.shape[0]//2, stack.shape[1]//2, stack.shape[2]//2 ]
        
        self.selected = []

        self.points = []

        self.labels_lut = np.array(colmap)

        self.zoom = 1
    
        gl = QtWidgets.QGridLayout()
        self.slider_x = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.slider_y = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.slider_z1 = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.slider_z2 = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
 
        self.slider_x.setMaximum(self.stack.shape[2]-1)
        self.slider_y.setMaximum(self.stack.shape[1]-1)
        self.slider_z1.setMaximum(self.stack.shape[0]-1)
        self.slider_z2.setMaximum(self.stack.shape[0]-1)

        self.slider_x.setValue(self.slice[2])
        self.slider_y.setValue(self.slice[1])
        self.slider_z1.setValue(self.slice[0])
        self.slider_z2.setValue(self.slice[0])

        self.slider_x.valueChanged.connect(self.x_slider_changed)
        self.slider_y.valueChanged.connect(self.y_slider_changed)
        self.slider_z1.valueChanged.connect(self.z1_slider_changed)
        self.slider_z2.valueChanged.connect(self.z2_slider_changed)

        

        self.setLayout(gl)
        gl.addWidget(self.slider_x, 1, 0)
        gl.addWidget(self.slider_y, 0, 1)
        gl.addWidget(self.slider_z1, 1, 2)
        gl.addWidget(self.slider_z2, 2, 1)

        print(stack.shape, self.spacing)
        
        r = self.spacing[2]/float(self.spacing[0])

        gl.setColumnStretch(0, stack.shape[2])
        gl.setColumnStretch(2, stack.shape[0]*r)
        gl.setRowStretch(0, stack.shape[1])
        gl.setRowStretch(2, stack.shape[0]*r)
        

        self.img_xy_w = QtWidgets.QScrollArea()
        self.img_xy = QtWidgets.QLabel()
        self.img_yz_w = QtWidgets.QScrollArea()
        self.img_yz = QtWidgets.QLabel()
        self.img_xz_w = QtWidgets.QScrollArea()
        self.img_xz = QtWidgets.QLabel()
        gl.addWidget(self.img_xy_w, 0, 0)
        self.img_xy_w.setWidget(self.img_xy)
        gl.addWidget(self.img_yz_w, 0, 2)
        self.img_yz_w.setWidget(self.img_yz)
        gl.addWidget(self.img_xz_w, 2, 0)        
        self.img_xz_w.setWidget(self.img_xz)

        self.img_xy.wheelEvent = self.wheel_xy
        self.img_yz.wheelEvent = self.wheel_yz
        self.img_xz.wheelEvent = self.wheel_xz


        self.img_xy.mousePressEvent = self.click_xy
        self.img_xy.mouseMoveEvent = self.click_xy

        self.img_yz.mousePressEvent = self.click_yz
        self.img_yz.mouseMoveEvent = self.click_yz

        self.img_xz.mousePressEvent = self.click_xz
        self.img_xz.mouseMoveEvent = self.click_xz
        
        self.update_labels()

    def set_label_opacity(self, v):
        self.label_opacity = v

    def click_xz(self, ev):
        p = ev.pos()
        w = self.img_xz.geometry().width()
        h = self.img_xz.geometry().height()
        self.set_slice([p.y()/float(h)*self.stack.shape[0], self.slice[1], p.x()/float(w)*self.stack.shape[2]])
        if (ev.buttons() & QtCore.Qt.LeftButton) and (ev.modifiers() & QtCore.Qt.ShiftModifier):
            if self.window is not None:
                self.window.drag_callback()
        
        self.update_labels()

    def wheel_yz(self, ev):
        try:
            d= ev.delta()/120
        except AttributeError:
            d= ev.angleDelta().y()/120
            print(ev.angleDelta())
        ev.accept()
        self.set_slice([self.slice[0], self.slice[1], max(min(self.slice[2]+d,self.stack.shape[2]-1),0)])
        
        self.update_labels()

    def wheel_xy(self, ev):
        try:
            d= ev.delta()/120
        except AttributeError:
            d= ev.angleDelta().y()/120
            print(ev.angleDelta())
        ev.accept()
 
            
        print('wheel', d)
        self.set_slice([max(min(self.slice[0]+d,self.stack.shape[0]-1),0), self.slice[1], self.slice[2]])
        self.update_labels()

    def wheel_xz(self, ev):
        try:
            d= ev.delta()/120
        except AttributeError:
            d= ev.angleDelta().x()/120
        ev.accept()
 
            
        self.set_slice([self.slice[0],  max(min(self.slice[1]+d,self.stack.shape[1]-1),0), self.slice[2]])
        self.update_labels()



    def click_xy(self, ev):
        p = ev.pos()
        w = self.img_xy.geometry().width()
        h = self.img_xy.geometry().height()
        self.set_slice([self.slice[0], p.y()/float(h)*self.stack.shape[1], p.x()/float(w)*self.stack.shape[2]])
        if (ev.buttons() & QtCore.Qt.LeftButton) and (ev.modifiers() & QtCore.Qt.ShiftModifier):
            if self.window is not None:
                self.window.drag_callback()

        self.update_labels()

    def click_yz(self, ev):
        p = ev.pos()
        w = self.img_yz.geometry().width()
        h = self.img_yz.geometry().height()
        self.set_slice([ p.x()/float(w)*self.stack.shape[0], p.y()/float(h)*self.stack.shape[1],self.slice[2]])
        if (ev.buttons() & QtCore.Qt.LeftButton) and (ev.modifiers() & QtCore.Qt.ShiftModifier):
            if self.window is not None:
                self.window.drag_callback()

        self.update_labels()


    def add_point(self, p):
        print('add point', p)
        self.points.append(p)
        self.update_labels()

    def get_point(self):
        return self.slice

    def update_colmap(self, colmap):
        self.labels_lut = np.array(colmap, dtype=float)

    def update(self, stack, img_stack):
        self.stack = stack
        if img_stack is not None:
            self.img_stack = img_stack
        else:
            self.img_stack = np.zeros(self.stack.shape)
        self.update_labels()
    
    def update_labels(self):

        if self.stack.dtype == np.float32:
            to_qimg = to_qimg_float
        else:
            to_qimg = to_qimg_lut


        u = self.img_stack[self.slice[0],:,:]
        

        def slice_points(points, i, dim, dim1, sc1, dim2, sc2, r=4):
            if points:
                print(points)
            pts = []
            s = np.array([sc1, sc2])
            for p, c in points:
                if abs(p[dim] - i)<r:
                    r2 = sqrt(r*r - (p[dim]-i)**2)
                    pts.append((np.array(p[[dim1, dim2]])*s, r2, c))
            return pts


        pts_xy = slice_points(self.points, self.slice[0], 0, 1, 1, 2, 1)

        im = to_qimg(self.stack[self.slice[0],:,:], u, self.slice[1], self.slice[2], self.labels_lut, pts_xy, label_opacity=self.label_opacity)
        pm = QtGui.QPixmap.fromImage(im)
        self.img_xy.setPixmap(pm)
        self.img_xy.setScaledContents(True)
        self.img_xy.resize(self.zoom*pm.size())

        r = self.spacing[2]/float(self.spacing[0])

        u = nd.zoom(self.img_stack[:,self.slice[1],:], [r, 1], order=0)

        pts_xz = slice_points(self.points, self.slice[1], 1, 0, r, 2, 1)

        im = to_qimg(nd.zoom(self.stack[:,self.slice[1],:], [r, 1], order=0), u, int(self.slice[0]*r), self.slice[2], self.labels_lut, pts_xz, label_opacity=self.label_opacity)
        pm = QtGui.QPixmap.fromImage(im)
        self.img_xz.setPixmap(pm)
        self.img_xz.setScaledContents(True)
        self.img_xz.resize(self.zoom*pm.size())
        

        r = self.spacing[2]/float(self.spacing[1])

        u = nd.zoom(self.img_stack[:,:,self.slice[2]].T, [1, r], order=0)

        pts_yz = slice_points(self.points, self.slice[2], 2, 1, 1, 0, r)

        im = to_qimg(nd.zoom(self.stack[:,:,self.slice[2]].T, [1, r], order=0), u, self.slice[1], int(self.slice[0]*r), self.labels_lut, pts_yz, label_opacity=self.label_opacity)
        pm = QtGui.QPixmap.fromImage(im)
        self.img_yz.setPixmap(pm)
        self.img_yz.setScaledContents(True)
        self.img_yz.resize(self.zoom*pm.size())

    def set_slice(self, slice_vals):
        slice_vals = [ min(max(0, slice_vals[i]), self.stack.shape[i]-1) for i in range(3) ]

        
        self.slice = np.array(slice_vals, dtype=np.int32)
        self.slider_x.blockSignals(True)
        self.slider_y.blockSignals(True)
        self.slider_z1.blockSignals(True)
        self.slider_z2.blockSignals(True)
        self.slider_x.setValue(slice_vals[2])
        self.slider_y.setValue(slice_vals[1])
        self.slider_z1.setValue(slice_vals[0])
        self.slider_z2.setValue(slice_vals[0])
        self.slider_x.blockSignals(False)
        self.slider_y.blockSignals(False)
        self.slider_z1.blockSignals(False)
        self.slider_z2.blockSignals(False)

    def x_slider_changed(self, value):
        self.slice[2] = value
        self.update_labels()

    def y_slider_changed(self, value):
        self.slice[1] = value
        self.update_labels()

    def z1_slider_changed(self, value):
        self.slider_z2.setValue(value)
        self.slice[0] = value
        self.update_labels()

    def z2_slider_changed(self, value):
        self.slider_z1.setValue(value)
        self.slice[0] = value
        
