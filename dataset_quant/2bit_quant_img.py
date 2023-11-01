import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.signal as signal
from sklearn.cluster import KMeans
import circle_fit as cf
from scipy import stats
import multiprocessing as mp
from torchvision import io
from torchvision import utils
import cv2
from PIL import Image

data_dir = '/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction/'
out_dir = '/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction-2bit/'
directories = os.listdir(data_dir)

def uniform_quant(img):
	img = torch.where(img >= 0b10100000, 0b11000000, img)
	img = torch.where((img >= 0b01100000) & (img < 0b10100000), 0b10000000, img)
	img = torch.where((img >= 0b00100000) & (img < 0b01100000), 0b01000000, img)
	img = torch.where(img < 0b00100000, 0b00000000, img)
	return img

def manual_quant(img):
	black_threshold = 42
	iris_threshold = 100
	skin_threshold = 210

	black_color = 0
	iris_color = 70
	skin_color = 160
	white_color = 255
	# first make the blacks 0
	img = torch.where(img < black_threshold, black_color, img)
	# make the iris color
	img = torch.where((img > black_threshold) & (img < iris_threshold), iris_color, img)
	# make the skin color
	img = torch.where((img > iris_threshold) & (img < skin_threshold), skin_color, img)
	# make the white color
	img = torch.where(img > skin_threshold, white_color, img)

def peak_quant(img):
	hist = Counter(img.flatten().tolist())
	x = list(range(255))
	y = [hist[i] if i in hist else 0 for i in x]

	peak_x, peak_y = signal.find_peaks(y, height=100, distance=30)
	peak_y = peak_y['peak_heights']
	if len(peak_y) > 3:
		ind = np.argpartition(peak_y, -3)[-3:]
		top4 = peak_x[ind].tolist()
		top4.append(255)
	else:
		top4 = peak_x.tolist()
		top4.append(255)
	top4 = np.sort(top4)

	bin_centers = (top4[1:]+top4[:-1])/2
	img_q = np.digitize(img, bin_centers, right=False)
	img_q = np.where(img_q == min(top4), 0, img_q)
	return img_q
	# plt.figure()
	# plt.imshow(img_q, cmap='gray')
	# plot_hist(img_q)

def kmeans_quant(img):
	temp = img.clone().numpy()
	temp = np.array([temp, np.zeros_like(temp), np.zeros_like(temp)])
	z = temp.reshape((-1,3))

	# convert to np.float32
	z = np.float32(z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 4
	ret,label,center=cv2.kmeans(z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

	# Convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((temp.shape))[0]
	return res2

def quantize(datatype_dir, directory, quant_scheme='kmeans'):
	for img_file in os.listdir(os.path.join(datatype_dir, directory)):
		img_path = os.path.join(datatype_dir, directory, img_file)
		img = io.read_image(img_path).to(torch.uint8)[0]

		match quant_scheme:
			case 'uniform':
				img = uniform_quant(img)

			case 'manual':
				img = manual_quant(img)

			case 'peaks':
				img = peak_quant(img)

			case 'kmeans':
				img = kmeans_quant(img)
				
		img = np.where(img == img.max(), 255, img).astype(np.uint8)

		out_temp_dir = os.path.join(out_dir, data_type, 'sequences', directory)
		print(img_file, out_temp_dir)
		if not os.path.isdir(out_temp_dir):
			os.system(f'mkdir -p {out_temp_dir}')
		out_img_path = os.path.join(out_temp_dir, img_file)

		img = Image.fromarray(img)
		img.save(out_img_path)

def job(id, cpus, datatype_dir, directories):
	i = id
	while i < len(directories):
		directory = directories[i]
		quantize(datatype_dir, directory)
		i += cpus


if __name__ == '__main__':
	cpus = 10

	for data_type in ['train', 'validation', 'test']:
		datatype_dir = os.path.join(data_dir, data_type, 'sequences')
		directories = os.listdir(datatype_dir)

		jobs = [mp.Process(target=job, args=(i, cpus, datatype_dir, directories)) for i in range(cpus)]

		for i in jobs:
			i.start()

		for i in jobs:
			i.join()
			
				

				# utils.save_image(img, out_img_path)
				# plt.imshow(img.reshape(img.shape[1:]), cmap='gray')
		# 		break
		# 	break
		# break