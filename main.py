from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import numpy as np
import cv2,math
import sys

class MainWindow(QMainWindow):
    widget=None
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ASI')
        self.resize(800,600)
        self.initUI()
        self.widget=Widget()
        self.setCentralWidget(self.widget)

    def initUI(self):
        menu=self.menuBar()
        file=menu.addMenu('&File')
        help=menu.addMenu('&?')
        new=QAction('&New..',self)
        new.setShortcut('Ctrl+N')
        file.addAction(new)
        open=QAction('&Open..',self)
        open.setShortcut('Ctrl+O')
        open.triggered.connect(self.open)
        file.addAction(open)
        quit=QAction('&Quit..',self)
        quit.setShortcut('Ctrl+Q')
        file.addAction(quit)

    def open(self):
        fname=QFileDialog().getOpenFileName(self,'Open Image','.\image',"Image files (*.jpg *.gif *.png)")
        self.widget.image.setPixmap(QPixmap(fname[0]))
        self.widget.path=fname[0]

class Widget(QWidget):
    image=None
    path=None
    def __init__(self):
        super().__init__()
        self.initUi()
        self.setStyleSheet("border:1px solid black;")
        self.asi=ASI(image=None)

    def initUi(self):
        _=['Convertion Image','Traitement Histogramme','Transformation Geometrique','Interpolation','Transformation Affine',
           'DFT','Seuillage','Detection Contour 1','Lissage','Detection Contour 2','Canny','Tranformee de Hough']
        layout=QHBoxLayout()
        label=QLabel()
        list=QListWidget(self)
        list.itemClicked.connect(self.listClicked)
        list.addItems(_)
        layout.addWidget(list,1)
        layout.addWidget(label,2)
        self.setLayout(layout)
        self.image=label

    def listClicked(self,item):
        image=cv2.imread(self.path)
        self.asi.image=image
        self.asi.grayimage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if item.text()=='Convertion Image':
            self.showImage(self.asi.convertionImage())
        if item.text()=='Traitement Histogramme':
            self.showHistogramme(self.asi.traitementhistogramme())
        if item.text()=='Transformation Geometrique':
            self.showImage(self.asi.transformationgeometrique())
        if item.text()=='Interpolation':
            self.showImage(self.asi.Interpolation())
        if item.text()=='Transformation Affine':
            self.showImage(self.asi.Transformationaffine())
        if item.text()=='DFT':
            self.showImage(self.asi.DFT())
        if item.text()=='Seuillage':
            self.showImage(self.asi.Seuillage())
        if item.text()=='Detection Contour 1':
            self.showImage(self.asi.Detectioncontour1())
        if item.text()=='Lissage':
            self.showImage(self.asi.Lissage())
        if item.text()=='Detection Contour 2':
            self.showImage(self.asi.Detectioncontour2())
        if item.text()=='Canny':
            self.showImage(self.asi.Canny())
        if item.text()=='Tranformee de Hough':
            self.showImage(self.asi.Tranformeehough())


    def showImage(self,images):
        fig=plt.figure()
        axes=[]
        for i in range(len(images)):
            axes.append(fig.add_subplot(2,int(len(images)/2),i+1))
            plt.imshow(images[i],cmap='gray')
        fig.tight_layout()
        plt.show()
    def showHistogramme(self,histogrammes):
        fig=plt.figure()
        axes=[]
        for i in range(len(histogrammes)):
            axes.append(fig.add_subplot(2,int(len(histogrammes)/2),i+1))
            if(len(histogrammes[i])==3):
                color = ('b', 'g', 'r')
                for j, col in enumerate(color):
                    plt.plot(histogrammes[i][j], color=col)
            else:
                plt.plot(histogrammes[i])
        fig.tight_layout()
        plt.show()
class ASI:

    def __init__(self,image):
        self.image=image
        self.grayimage=None

    def convertionImage(self):
        def bincv():
            return cv2.threshold(cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)[1]
        def ngcv():
            return cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        def hsvcv():
            return cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
        def ycrcbcv():
            return cv2.cvtColor(self.image,cv2.COLOR_BGR2YCrCb)
        def bin():
            image=ng()
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if(image[i][j]>128):
                        image[i][j]=255
                    else:
                        image[i][j]=0
            return image
        def ng():
            image=np.zeros(self.image.shape[:2]).astype(int)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    image[i][j]=(self.image[i][j][0]*0.3)+(self.image[i][j][1]*0.59)+(self.image[i][j][2]*0.11)
            return image
        def hsv():
            image=np.zeros(self.image.shape).astype(int)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    r=self.image[i][j][0]/255.0
                    g=self.image[i][j][1]/255.0
                    b=self.image[i][j][2]/255.0
                    M = max(r,g,b)
                    m= min(r,g,b)
                    c = M - m
                    if c==0:
                        h=0
                    elif M == r:
                        h = (60 * ((g - b) / c) + 360) / 360
                    elif M == g:
                        h = (60 * ((b - r) / c) + 120) / 360
                    elif M == b:
                        h = (60 * ((r - g) / c) + 240) / 360
                    if M == 0:
                        s = 0
                    else:
                        s = (c / M) * 255
                    v = M * 255
                    h=h*180
                    image[i][j]=[h,s,v]
            return image
        def ycrcb():
            image=np.zeros(self.image.shape).astype(int)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    y=0.3*self.image[i][j][2]+0.6*self.image[i][j][1]+0.1*self.image[i][j][0]
                    image[i][j]=[y,128+0.5*(self.image[i][j][2]-y),128+0.5*(self.image[i][j][0]-y)]
            return image
        return bincv(),ngcv(),hsvcv(),ycrcbcv(),bin(),ng(),hsv(),ycrcb()
    def traitementhistogramme(self):
        def histogrammegraycv():
           return cv2.calcHist([self.grayimage],[0], None, [256], [ 0, 256])
        def histogrammecolorcv():
            _=[]
            for i in range(3):
                _.append(cv2.calcHist([self.image],[i], None, [256], [ 0, 256]))
            return _
        def histogrammeequalgraycv():
             return cv2.calcHist([cv2.equalizeHist(self.grayimage)],[0], None, [256], [ 0, 256])
        def histogrammenormalcv():
            return cv2.calcHist([cv2.normalize(self.grayimage,np.zeros(self.grayimage.shape),0,255,cv2.NORM_MINMAX)],[0], None, [256], [ 0, 256])
        def histogrammegray():
            hist=np.zeros((256)).astype(int)
            for i in range(self.grayimage.shape[0]):
                for j in range(self.grayimage.shape[1]):
                    hist[self.grayimage[i,j]]+=1
            return hist
        def histogrammecolor():
            _=[np.zeros((256)).astype(int)for i in range(3)]
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    _[0][self.image[i,j][0]]+=1
                    _[1][self.image[i,j][1]]+=1
                    _[2][self.image[i,j][2]]+=1
            return _
        def histogrammecumlegray():
            hist=histogrammegray()
            for i in range(1,len(hist)):
                hist[i]+=hist[i-1]
            return hist
        def histogrammecumlecolor():
            _=histogrammecolor()
            for i in range(len(_)):
                for j in range(1,len(_[i])):
                    _[i][j]+=_[i][j-1]
            return _
        def histogrammeequalgray():
            hist=histogrammecumlegray()
            image=np.zeros(self.grayimage.shape).astype(int)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j]=int(255*hist[self.grayimage[i][j]]/(image.shape[0]*image.shape[1]))
            hist=np.zeros((256)).astype(int)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    hist[image[i,j]]+=1
            return hist
        def histogrammenormal():
            image=np.zeros(self.grayimage.shape).astype(int)
            max_value=np.max(self.grayimage)
            min_value=np.min(self.grayimage)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j]=int((255/(max_value-min_value))*(self.grayimage[i][j]-min_value))
            hist=np.zeros((256)).astype(int)
            for i in range(self.grayimage.shape[0]):
                for j in range(self.grayimage.shape[1]):
                    hist[image[i,j]]+=1
            return hist
        def histogramme_distance(hist1,hist2):
            somme=0
            for i in range(256):
                somme+=np.abs(hist1[i]-hist2[i])
            return somme

        return histogrammegraycv(),histogrammecolorcv(),histogrammeequalgraycv(),histogrammenormalcv(),histogrammecumlegray(),histogrammegray(),histogrammecolor(),histogrammeequalgray(),histogrammenormal(),histogrammecumlecolor()

    def transformationgeometrique(self):

        def translationcv(m=np.float32([[1, 0, 80],[ 0, 1, 80]])):
            return cv2.warpAffine(self.grayimage,m,self.grayimage.shape)

        def rotationcv(angle,cx,cy):
            return cv2.warpAffine(self.grayimage,cv2.getRotationMatrix2D((cx,cy), angle , 1),self.grayimage.shape)

        def zoomcv():
            return cv2.resize(self.grayimage,(self.grayimage.shape[0]*2,self.grayimage.shape[1]*2))

        def cisaillementcv(m=np.float32([[1,0.5, 0],[ 0, 1, 1]])):
            return cv2.warpAffine(self.grayimage,m,self.grayimage.shape)

        def translation(x,y):
            image=np.zeros(self.grayimage.shape).astype(int)
            for i in range(self.grayimage.shape[0] - x):
                for j in range(self.grayimage.shape[1] - y):
                    image[i+x][j+y] = self.grayimage[i][j]
            return image

        def rotation(angle,cx,cy):
            image=np.zeros(self.grayimage.shape).astype(int)
            alpha=math.radians(angle)
            for i in range(self.grayimage.shape[0]):
                for j in range(self.grayimage.shape[1]):
                    x=round(cx + ((i - cx)*math.cos(alpha)) +  ((j - cy)*math.sin(alpha)))
                    y=round(cy - ((i - cx)*math.sin(alpha)) +  ((j - cy)*math.cos(alpha)))
                    if (x<len(image) and -1<x and y<len(image[0]) and -1<y):
                        image[i][j]=self.grayimage[x][y]
            return image

        def round(x):
                if(   0.5 < x %1 ):
                    return math.ceil(x)
                return math.floor(x)

        def zoom(h,w):
            image=np.zeros((h,w))
            x,y=h/self.grayimage.shape[0],w/self.grayimage.shape[0]
            for i in range(h):
                for j in range(w):
                    image[i][j]=self.grayimage[round(i/x)][round(j/y)]
            return image

        def cisaillement(shx,shy):
            image=np.zeros(self.grayimage.shape).astype(int)
            for i in range(self.grayimage.shape[0]):
                for j in range(self.grayimage.shape[1]):
                    x=round((i+(shx * j)))
                    y=round((j+(shy * i)))
                    if (x<len(image) and -1<x and y<len(image[0]) and -1<y):
                        image[x][y]=self.grayimage[i][j]
            return image

        return translationcv(),rotationcv(20,20,20),zoomcv(),cisaillementcv(),translation(80,80),rotation(20,20,20),zoom(1024,1024),cisaillement(0,0.5)

    def Interpolation(self):
        def rotation(angle,cx,cy):
            image=np.zeros(self.grayimage.shape).astype(int)
            alpha=math.radians(angle)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    x=cx + ((i - cx)*math.cos(alpha)) +  ((j - cy)*math.sin(alpha))
                    y=cy - ((i - cx)*math.sin(alpha)) +  ((j - cy)*math.cos(alpha))
                    x_min=math.floor(x)
                    x_max=math.ceil(x)
                    y_min=math.floor(y)
                    y_max=math.ceil(y)
                    dx=x-x_min
                    dy=y-y_min
                    if  x_max == len(image):
                        x_max-=1
                    if  y_max == len(image[0]):
                        y_max-=1
                    if (x<len(image) and -1<x and y<len(image[0]) and -1<y):
                        image[i][j]=(((1-dx)*(1-dy)*self.grayimage[x_min][y_min])+((1-dx)*(dy)*self.grayimage[x_min][y_max])+((dx)*(1-dy)*self.grayimage[x_max][y_min])+((dx)*(dy)*self.grayimage[x_max][y_max]))
            return image
        
        def zoom( h, w):
            image=np.zeros((h,w)).astype(int)
            x=h/self.grayimage.shape[0]
            y=w/self.grayimage.shape[1]
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    x_min=math.floor(i/x)
                    x_max=math.ceil(i/x)
                    y_min=math.floor(j/y)
                    y_max=math.ceil(j/y)
                    dx = (i/x) - x_min
                    dy = (j/y) - y_min
                    if  x_max == self.grayimage.shape[0]:
                        x_max-=1
                    if  y_max == self.grayimage.shape[1]:
                        y_max-=1
                    image[i][j]= (((1-dx) * (1-dy) * self.grayimage[x_min][y_min]) + ((1 - dx) * (dy) * self.grayimage[x_min][y_max]) + ((dx) * (1 - dy) * self.grayimage[x_max][y_min]) + ((dx) * (dy) * self.grayimage[x_max][y_max]))
            return image
        return rotation(20,20,20),zoom(1024,1024)

    def Transformationaffine(self):
        def tcv(m1 = np.float32([[40,40],[200,40],[40,200]]),m2 = np.float32([[10,100],[200,40],[100,250]])):
            transform_matrix=cv2.getAffineTransform(m1,m2)
            return cv2.warpAffine(self.grayimage,transform_matrix,self.grayimage.shape)

        def t(matrix):
            image=np.zeros(self.grayimage.shape).astype(int)
            matrix_sqr=np.vstack([matrix,[0,0,1]])
            matrix_inv=np.linalg.inv(matrix_sqr)
            for i in range(len(image)):
                for j in range(len(image[0])):
                    x=int(round(matrix_inv[0][0]*i + matrix_inv[0][1]*j + matrix_inv[0][2]))
                    y=int(round(matrix_inv[1][0]*i + matrix_inv[1][1]*j + matrix_inv[1][2]))
                    if (x<len(image) and -1<x and y<len(image[0]) and -1<y):
                        image[j][i]=self.grayimage[y][x]
            return image
        def tra():
            matrix = np.array([10,100,200,40,100,250])
            A = np.array([[40,40,1,0,0,0],[0,0,0,40,40,1],[200,40,1,0,0,0],[0,0,0,200,40,1],[40,200,1,0,0,0],[0,0,0,40,200,1]])
            A_inverse=np.linalg.inv(A)
            X=np.dot(A_inverse,matrix)
            X=np.reshape(X,(2,3))
            return t(X)

        return tcv(),tra()

    def DFT(self):
        def dftcv():
            h= cv2.getOptimalDFTSize(self.grayimage.shape[0])
            w= cv2.getOptimalDFTSize(self.grayimage.shape[1])
            padding = cv2.copyMakeBorder(self.grayimage, 0, h - len(self.image), 0, w - len(self.image[0]), cv2.BORDER_CONSTANT, value=[0, 0, 0])
            planes = [np.float32(padding), np.zeros(padding.shape, np.float32)]
            complexI = cv2.merge(planes)
            cv2.dft(complexI, complexI)
            cv2.split(complexI, planes)
            cv2.magnitude(planes[0], planes[1], planes[0])
            magI = planes[0]
            matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
            cv2.add(matOfOnes, magI, magI)
            cv2.log(magI, magI)
            magI_rows, magI_cols = magI.shape
            magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
            cx = int(magI_rows/2)
            cy = int(magI_cols/2)
            q0 = magI[0:cx, 0:cy]
            q1 = magI[cx:cx+cx, 0:cy]
            q2 = magI[0:cx, cy:cy+cy]
            q3 = magI[cx:cx+cx, cy:cy+cy]
            tmp = np.copy(q0)
            magI[0:cx, 0:cy] = q3
            magI[cx:cx + cx, cy:cy + cy] = tmp
            tmp = np.copy(q1)
            magI[cx:cx + cx, 0:cy] = q2
            magI[0:cx, cy:cy + cy] = tmp
            cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX)
            return magI

        def dft():
            img=cv2.resize(self.grayimage,(50,50))
            image=np.zeros(img.shape)
            for k in range(len(img)):
                for l in range(len(img[0])):
                    s1,s2=0,0
                    for i in range(len(img)):
                        for j in range(len(img[0])):
                            s1+=(img[i][j]*(math.cos(-2*(math.pi)*((k*i/len(img)) + (l*j/len(img[0]))))))
                            s2+=((img[i][j]*(math.sin(-2*(math.pi)*((k*i/len(img)) + (l*j/len(img[0])))))))
                    image[k][l]=math.log(math.sqrt((s2) ** 2  +  (s1) ** 2)+1)
            cx = int(image.shape[0]/2)
            cy = int(image.shape[1]/2)
            q0 = image[0:cx, 0:cy]
            q1 = image[cx:cx+cx, 0:cy]
            q2 = image[0:cx, cy:cy+cy]
            q3 = image[cx:cx+cx, cy:cy+cy]
            tmp = np.copy(q0)
            image[0:cx, 0:cy] = q3
            image[cx:cx + cx, cy:cy + cy] = tmp
            tmp = np.copy(q1)
            image[cx:cx + cx, 0:cy] = q2
            image[0:cx, cy:cy + cy] = tmp
            cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
            return image
        return dftcv(),dft()

    def Convolution(self,img,m,t):
        image=np.zeros(img.shape).astype(int)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                s=0
                b=False
                for k in range(m.shape[0]):
                    for l in range(m.shape[1]):
                        try:
                            if (i-m.shape[0]//2)+k >= 0 and (j-m.shape[1]//2)+l >= 0:
                                s+=m[k][l]*img[(i-m.shape[0]//2)+k][(j-m.shape[1]//2)+l]
                            else:
                                b=True
                        except:
                            b=True
                            s+=0
                if b:
                   image[i][j]=img[i][j]
                else:
                    image[i][j]=s/t
        return image

    def Seuillage(self):
        def seucv():
            return cv2.threshold(self.grayimage,70,150,cv2.THRESH_BINARY)[1]

        def binary(img,max,min):
            image=np.zeros(img.shape).astype(int)
            tab=[-1,0,1]
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] > max:
                        image[i][j]=1
                    elif img[i][j]<min:
                        image[i][j]=0
                    else:
                        s=0
                        for k in tab:
                            x=i+k
                            for l in tab:
                                y=j+l
                                if(x<len(img) and -1<x and y<len(img[0]) and -1<y and max < img[i][j]):
                                    s=1
                        image[i][j]=s
            return image

        return seucv(),binary(self.grayimage,70,150)

    def Detectioncontour1(self):
        def naif(m=np.array([[0,0,0],[-1,1,0],[0,0,0]]),m1=np.array([[0,-1,0],[0,1,0],[0,0,0]])):
            naifx=self.Convolution(self.grayimage,m,1)
            naify=self.Convolution(self.grayimage,m1,1)
            return np.array([[np.sqrt((naifx[i][j]**2)+(naify[i][j]**2)) for j in range(self.grayimage.shape[1])]for i in range(self.grayimage.shape[0])])

        def sobel(m=np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),m1=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])):
                 sobelx=self.Convolution(self.grayimage,m,4)
                 sobely=self.Convolution(self.grayimage,m1,4)
                 return np.array([[np.sqrt((sobelx[i][j]**2)+(sobely[i][j]**2)) for j in range(self.grayimage.shape[1])]for i in range(self.grayimage.shape[0])])

        def roberts(m=np.array([[0,0,0],[0,1,0],[0,0,-1]]),m1=np.array([[0,0,0],[0,0,1],[0,-1,0]])):
            robertsx=self.Convolution(self.grayimage,m,1)
            robertsy=self.Convolution(self.grayimage,m1,1)
            return np.array([[np.sqrt((robertsx[i][j]**2)+(robertsy[i][j]**2)) for j in range(self.grayimage.shape[1])]for i in range(self.grayimage.shape[0])])

        def prewitt(m=np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),m1=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])):
                 prewittx=self.Convolution(self.grayimage,m,3)
                 prewitty=self.Convolution(self.grayimage,m1,3)
                 return np.array([[np.sqrt((prewittx[i][j]**2)+(prewitty[i][j]**2)) for j in range(self.grayimage.shape[1])]for i in range(self.grayimage.shape[0])])

        return naif(),sobel(),roberts(),prewitt()

    def Lissage(self):
        def Gaussien(m=np.array([[1,1,1],[1,8,1],[1,1,1]])):
            return self.Convolution(self.grayimage,m,16)
        def Moyen(m=np.array([[1,1,1],[1,1,1],[1,1,1]])):
            return self.Convolution(self.grayimage,m,9)

        return cv2.blur(self.grayimage,(3,3)),cv2.GaussianBlur(self.grayimage,(3,3),3),Gaussien(),Moyen()

    def Detectioncontour2(self):
        def Laplace(m=np.array([[1,1,1],[1,-8,1],[1,1,1]])):
            return self.Convolution(self.grayimage,m,1)

        def Log(m=np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])):
            return self.Convolution(self.grayimage,m,1)

        def Dog(m=np.array([[1,1,1],[1,-8,1],[1,1,1]]),m1=np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])):
            imf=self.Convolution(self.grayimage,m,1)
            ims=self.Convolution(self.grayimage,m1,1)
            image=np.zeros(self.grayimage.shape).astype(int)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j]=np.abs(imf[i][j] - ims[i][j])
            return image
        return np.uint8(np.absolute(cv2.Laplacian(self.grayimage,cv2.CV_64F))),np.uint8(np.absolute(cv2.Laplacian(cv2.GaussianBlur(self.grayimage,(3,3),3),cv2.CV_64F))),Dog(),np.uint8(np.absolute(Laplace())),np.uint8(np.absolute(Log())),Dog()
    def Canny(self):
        def sobel(m=np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),m1=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])):
                 direction=np.zeros(self.grayimage.shape).astype(float)
                 lissage=self.Convolution(self.grayimage,np.array([[1,1,1],[1,1,1],[1,1,1]]),9)
                 sobelx=self.Convolution(lissage,m,4)
                 sobely=self.Convolution(lissage,m1,4)
                 for i in range(direction.shape[0]):
                     for j in range(direction.shape[1]):
                         if sobelx[i][j]==0:
                             direction[i][j]=0
                         else:
                             a=math.atan2(sobely[i][j],sobelx[i][j])
                             direction[i][j]=(math.degrees(a)+180) % 180
                 return np.array([[np.sqrt((sobelx[i][j]**2)+(sobely[i][j]**2)) for j in range(self.grayimage.shape[1])]for i in range(self.grayimage.shape[0])]),direction
        def nms(img,direction):
            image=np.zeros(img.shape)
            h,w=img.shape[0]-1,img.shape[1]-1
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if direction[i][j] <= 22.5  or 157.5 < direction[i][j]:
                        if img[i][j] < max(img[min(h,i+1)][max(0,j-1)],img[max(0,i-1)][min(w,j+1)]):
                            image[i][j]=0
                        else:
                            image[i][j]=img[i][j]
                    elif direction[i][j] <= 22.5  or 67.5 < direction[i][j]:
                        if img[i][j] < max(img[min(h,i+1)][j],img[max(0,i-1)][j]):
                            image[i][j]=0
                        else:
                            image[i][j]=img[i][j]
                    elif direction[i][j] <= 67.5  or 112.5 < direction[i][j]:
                        if img[i][j] < max(img[i][min(w,j+1)],img[i][max(0,j-1)]):
                            image[i][j]=0
                        else:
                            image[i][j]=img[i][j]
                    elif direction[i][j] <= 112.5  or 157.5 < direction[i][j]:
                        if img[i][j] < max(img[min(h,i+1)][min(w,j+1)],img[max(0,i-1)][max(0,j-1)]):
                            image[i][j]=0
                        else:
                            image[i][j]=img[i][j] 
            return image
        def binary(img,max,min):
            image=np.zeros(img.shape).astype(int)
            tab=[-1,0,1]
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] > max:
                        image[i][j]=1
                    elif img[i][j]<min:
                        image[i][j]=0
                    else:
                        s=0
                        for k in tab:
                            x=i+k
                            for l in tab:
                                y=j+l
                                if(x<len(img) and -1<x and y<len(img[0]) and -1<y and max < img[i][j]):
                                    s=1
                        image[i][j]=s
            return image
        sobel,direction=sobel()
        imagenms=nms(sobel,direction)
        return cv2.Canny(self.grayimage,100,200),binary(imagenms,35,10)

    def Tranformeehough(self):
        cany=cv2.Canny(self.grayimage,50,150,3)
        def droitescv():
            image=self.image.copy()
            lines=cv2.HoughLines(cany,1,np.pi/180,100)
            for r,t in lines[0]:
                a,b=np.cos(t),np.sin(t)
                x,y=a*r,b*r
                x1,y1=int(x+1000*(-b)),int(y+1000*(a))
                x2,y2=int(x-1000*(-b)),int(y-1000*(a))
                cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
            return image
        def cerclescv():
            image=self.image.copy()
            circles = cv2.HoughCircles()

        return droitescv(),droitescv()


if __name__ == '__main__':
    app=QApplication(sys.argv)
    main_window=MainWindow()
    main_window.show()
    sys.exit(app.exec_())
