import torch
import torch.nn as nn
import numpy as np


# нормировки полиномов, приняиые в Optical Society of America
Z_OSA = np.array([1,
                 4, 4,
                 6, 3, 6,
                 8, 8, 8, 8,
                 10, 10, 5, 10, 10,
                 12, 12, 12, 12, 12, 12,
                 14, 14, 14, 7, 14, 14, 14])**.5


# Разреженная нормировка изображений
class SparseNormilize(nn.Module):
    def __init__(self, mean, std, eps):
        super().__init__()
        dim = len(mean)
        self.mean = torch.Tensor(mean)[:, None, None]
        self.std = torch.Tensor(std)[:, None, None]#.view(dim, 1, 1)
        self.eps = eps
    def forward(self, X):
        nz = (X > self.eps)
        return (X - self.mean * nz) / self.std

# позиция наиболее яркого пикселя
def center_max(I):
	imax = np.argmax(I)
	return imax % I.shape[1], imax // I.shape[1]


# позиция центра масс изображеня
def center_mean(I):
	ii = np.arange(0, I.shape[1])
	jj = np.arange(0, I.shape[0])
	s = np.sum(I)
	return int(np.sum(jj * I.T) / s), int(np.sum(ii * I) / s)

# обрезка
def crop(I,rc,w):
	return I[rc[0] - w:rc[0] + w, rc[1] - w:rc[1] + w]

# обрезка с дополнением нулями
def crop_padding(I,rc,w,pad=0.0):
    tmp=I.copy()
    rc=np.array(rc)
    if rc[0]-w>=0:
        tmp=tmp[rc[0]-w:,:]
    else:
        tmp=np.pad(tmp,((w-rc[0],0),(0,0)))
    rc[0]=w
    if rc[0]+w<tmp.shape[0]:
        tmp=tmp[:2*w,:]
    else:
        tmp=np.pad(tmp,((0,2*w-tmp.shape[0]),(0,0)))
    if rc[1]-w>=0:
        tmp=tmp[:,rc[1]-w:]
    else:
        tmp=np.pad(tmp,((0,0),(w-rc[1],0)))
    rc[1]=w
    if rc[1]+w<tmp.shape[1]:
        tmp=tmp[:,:2*w]
    else:
        tmp=np.pad(tmp,((0,0),(0,2*w-tmp.shape[1])))
    return tmp

# fft и ifft. Версия из scipy как будто выдаёт неправильную фазу
def Fi(I):
	ii = np.linspace(0, I.shape[0] - 1, I.shape[0])
	jj = np.linspace(0, I.shape[1] - 1, I.shape[1])
	x,y = np.meshgrid(ii, jj, sparse=True)
	phase_factor = np.exp(-1j * np.pi * (x + y))

	ift = np.fft.ifft2(phase_factor * I)
	return np.array( phase_factor * ift).reshape(I.shape)
def F(I):
	ii = np.linspace(0, I.shape[0] - 1, I.shape[0])
	jj = np.linspace(0, I.shape[1] - 1, I.shape[1])
	x,y = np.meshgrid(ii, jj, sparse=True)
	phase_factor = np.exp(-1j * np.pi * (x + y))

	ift = np.fft.fft2(phase_factor * I)
	return np.array( phase_factor * ift).reshape(I.shape)


# Расчёт диаметров по определённому уровню
def FWHM(Y,level=1/np.e**2):
    half_max = np.max(Y) *level
    d = np.where(np.sign(Y-half_max)>0)[0]
    if len(d)<=2:
        return -1
    return d[-1]-d[0]

#  Полиномы Цернике. Код сгенерирован
def Z_gen(i,X,Y):
	if i==0:
		return np.ones(X.shape)
	elif i==1:
		return Y
	elif i==2:
		return X
	elif i==3:
		return 2.0*X**1*Y**1
	elif i==4:
		return 2.0*X**2+2.0*Y**2-1.0
	elif i==5:
		return X**2-Y**2
	elif i==6:
		return 3.0*X**2*Y**1-Y**3
	elif i==7:
		return 3.0*X**2*Y**1+3.0*Y**3-2.0*Y
	elif i==8:
		return 3.0*X**3+3.0*X**1*Y**2-2.0*X
	elif i==9:
		return X**3-3.0*X**1*Y**2
	elif i==10:
		return 4.0*X**3*Y**1-4.0*X**1*Y**3
	elif i==11:
		return 8.0*X**3*Y**1+8.0*X**1*Y**3-6.0*X**1*Y**1
	elif i==12:
		return 6.0*X**4+12.0*X**2*Y**2+6.0*Y**4-6.0*X**2-6.0*Y**2+1.0
	elif i==13:
		return 4.0*X**4+4.0*X**2*Y**2-3.0*X**2-4.0*X**2*Y**2-4.0*Y**4+3.0*Y**2
	elif i==14:
		return X**4-6.0*X**2*Y**2+Y**4
	elif i==15:
		return 5.0*X**4*Y**1-10.0*X**2*Y**3+Y**5
	elif i==16:
		return 15.0*X**4*Y**1+15.0*X**2*Y**3-12.0*X**2*Y**1-5.0*X**2*Y**3-5.0*Y**5+4.0*Y**3
	elif i==17:
		return 10.0*X**4*Y**1+20.0*X**2*Y**3+10.0*Y**5-12.0*X**2*Y**1-12.0*Y**3+3.0*Y
	elif i==18:
		return 10.0*X**5+20.0*X**3*Y**2+10.0*X**1*Y**4-12.0*X**3-12.0*X**1*Y**2+3.0*X
	elif i==19:
		return 5.0*X**5+5.0*X**3*Y**2-4.0*X**3-15.0*X**3*Y**2-15.0*X**1*Y**4+12.0*X**1*Y**2
	elif i==20:
		return X**5-10.0*X**3*Y**2+5.0*X**1*Y**4
	elif i==21:
		return 6.0*X**5*Y**1-20.0*X**3*Y**3+6.0*X**1*Y**5
	elif i==22:
		return 24.0*X**5*Y**1+24.0*X**3*Y**3-20.0*X**3*Y**1-24.0*X**3*Y**3-24.0*X**1*Y**5+20.0*X**1*Y**3
	elif i==23:
		return 30.0*X**5*Y**1+60.0*X**3*Y**3+30.0*X**1*Y**5-40.0*X**3*Y**1-40.0*X**1*Y**3+12.0*X**1*Y**1
	elif i==24:
		return 20.0*X**6+60.0*X**4*Y**2+60.0*X**2*Y**4+20.0*Y**6-30.0*X**4-60.0*X**2*Y**2-30.0*Y**4+12.0*X**2+12.0*Y**2-1.0
	elif i==25:
		return 15.0*X**6+30.0*X**4*Y**2+15.0*X**2*Y**4-20.0*X**4-20.0*X**2*Y**2+6.0*X**2-15.0*X**4*Y**2-30.0*X**2*Y**4-15.0*Y**6+20.0*X**2*Y**2+20.0*Y**4-6.0*Y**2
	elif i==26:
		return 6.0*X**6+6.0*X**4*Y**2-5.0*X**4-36.0*X**4*Y**2-36.0*X**2*Y**4+30.0*X**2*Y**2+6.0*X**2*Y**4+6.0*Y**6-5.0*Y**4
	elif i==27:
		return X**6-15.0*X**4*Y**2+15.0*X**2*Y**4-Y**6
	else:
		return np.zeros(X.shape)