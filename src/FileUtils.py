import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom
from torchvision.transforms import CenterCrop

from MathUtils import center_mean, crop_padding

# Функция для чтения показаний ДВФ
def load_coeffs(path, n = 10):
	wfs_data = pd.read_csv(path, sep = '\t')
	zernike_columns = wfs_data.columns[17:35]
	wfs_data = wfs_data[zernike_columns]
	return wfs_data.iloc[n].to_numpy()

# Функция для чтения распределения интенсивности,
# сохранённого в формате png
def read_png(path, dtype = 'uint16', channel = 0):
	img = Image.open(path)
	img.load()
	img = np.asarray(img,dtype=dtype)
	if len(img.shape) == 3:
		I = img[:, :, channel]
	else:
		I = img
	return I


# Для подоготовки изображения к подаче на вход модели
def prepare_exp_image(path, dxz, noise = 0, dij = (0, 0), padding = 50, save_width = 184//2):
	dx_cam_ = 2.4e-3
	di, dj = dij

	xs, xe = int((4.5-di) / dx_cam_), int((9.5 - di) / dx_cam_)
	ys, ye = int((1 - dj) / dx_cam_), int((6 - dj) / dx_cam_)

	X = read_png(path)[ys:ye, xs:xe] >> 4
	X = X.astype('float32')
	X = X - noise
	X = X * (X > 0)
	X /= np.max(X)
	X = zoom(X, dx_cam_ / dxz)
	xc, yc = center_mean(X)
	X = crop_padding(X, (xc, yc), padding)
	X = torch.Tensor(X)
	X = X * (X>0)
	X = np.sqrt(X)
	X = CenterCrop(save_width * 2)(X)
	return X

