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
from octree_quantizer import OctreeQuantizer, Color

data_dir = '/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction/'
out_dir = '/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction-1bit-otsu/'
directories = os.listdir(data_dir)


class KDTree:
    def __init__(self, root, left = None, right = None):
        self.root = root
        self.left = left
        self.right = right
    
    def set_left(self, left):
        self.left = left
    
    def set_right(self, right):
        self.right = right

    def count_leaves(self):
        # print(self.root)
        if self.root is None:
            return 0
        if self.left is None and self.right is None:
            # print(self.root.thresh)
            return 1
        left = self.left.count_leaves()
        right = self.right.count_leaves()
        return left + right

class Partition:
    def __init__(self,img):
        self.image = img
        self.left = None
        self.right = None
        self.thresh = None
        self.priority = None

    def find_optimal_split(self):
        var = np.var(self.image)

        # split pixels according to partition otsu
        self.thresh = filters.threshold_otsu(self.image)
        self.left = np.asarray([i for i in self.image if  i <= self.thresh])
        self.right = np.asarray([i for i in self.image if i > self.thresh])

        left_var = np.var(self.left)
        right_var = np.var(self.right)
        self.priority = var - (left_var + right_var)

    def split(self):
        return Partition(self.left), Partition(self.right)

def min_var_quant(img, bits):

    leaves = 2**bits
    root_part = Partition(img.flatten())
    root = KDTree(root_part)
    root_part.find_optimal_split()

    q = PriorityQueue()
    q.put((-1 * root_part.priority, [root, root_part]))

    while(root.count_leaves() != leaves):
        item = q.get()[1]
        node = item[0]
        part = item[1]

        left_part, right_part = part.split()

        left_node = KDTree(left_part)
        right_node = KDTree(right_part)

        node.set_left(left_node)
        node.set_right(right_node)

        right_part.find_optimal_split()
        q.put((-1 * right_part.priority, [right_node, right_part]))

        left_part.find_optimal_split()
        q.put((-1 * left_part.priority, [left_node , left_part]))

    def thresh(elem, root):
        if root.left is None and root.right is None:
            return root.root.thresh
        elif elem <= root.root.thresh:
            return thresh(elem, root.left)
        else:
            return thresh(elem,root.right)
    
    thresh_img = np.vectorize(thresh)(img, root)
    

    return thresh_img

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

def kmeans_quant(img, bits=2):
	temp = img.clone().numpy()
	temp = np.array([temp, np.zeros_like(temp), np.zeros_like(temp)])
	z = temp.reshape((-1,3))

	# convert to np.float32
	z = np.float32(z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 2**bits
	ret,label,center=cv2.kmeans(z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

	# Convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((temp.shape))[0]
	return res2

def octree_quant(image, bits=2):
	image = Image.fromarray(image.numpy())
	pixels = image.load()
	width, height = image.size

	octree = OctreeQuantizer()

	# add colors to the octree
	for j in range(height):
		for i in range(width):
			octree.add_color(Color(red = int(pixels[i, j])))

	# 256 colors for 8 bits per pixel output image
	palette = octree.make_palette(2**bits)

	# save output image
	out_image = Image.new('RGB', (width, height))
	out_pixels = out_image.load()
	for j in range(height):
		for i in range(width):
			index = octree.get_palette_index(Color(red = int(pixels[i, j])))
			color = palette[index]
			out_pixels[i, j] = (int(color.red), int(color.green), int(color.blue))
	out_image = np.array(out_image)[:, :, 0]
	out_image = np.where(out_image == out_image.max(), 255, out_image).astype(np.uint8)
	out_image = np.where(out_image == out_image.min(), 0, out_image).astype(np.uint8)
	return torch.tensor(out_image).to(torch.uint8)

def canny_quant(img):
	img = cv2.GaussianBlur(img.numpy(), (11,11), 0)
	img = cv2.Canny(image=img, threshold1=15, threshold2=22) 
	return torch.tensor(img).to(torch.uint8)

def compute_otsu_criteria(im, th):
    """Otsu's method to compute criteria."""
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1

def otsu_1bit(img):
	im = img.clone().numpy()
	threshold_range = range(np.max(im)+1)
	criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
	best_threshold = threshold_range[np.argmin(criterias)]
	otsu = np.where(im > best_threshold, 255, 0)
	edges = canny_quant(img)
	final_img = edges + otsu
	final_img = np.where(final_img >= 255, 255, 0)
	return torch.tensor(final_img).to(torch.uint8)


def quantize(datatype_dir, directory, quant_scheme='otsu_1bit', bits=2):
	img_files = sorted(os.listdir(os.path.join(datatype_dir, directory)))
	
	for img_file in img_files:
		out_temp_dir = os.path.join(out_dir, data_type, 'sequences', directory)
		print(img_file, out_temp_dir)
		if not os.path.isdir(out_temp_dir):
			os.system(f'mkdir -p {out_temp_dir}')
		out_img_path = os.path.join(out_temp_dir, img_file)
		if os.path.isfile(out_img_path):
			try:
				img = Image.open(out_img_path)
				img.verify()
				img.close()
				print('skipping', out_img_path)
				continue
			except:
				print('corrupt', out_img_path)
				os.system(f'rm {out_img_path}')

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
				img = kmeans_quant(img, bits=bits)

			case 'octree':
				img = octree_quant(img, bits=bits)

			case 'edge':
				img = canny_quant(img)

			case 'min_var_quant':
				img = min_var_quant(img, bits=bits)
			case 'otsu_1bit':
				img = otsu_1bit(img)
				
		img = np.where(img == img.max(), 255, img).astype(np.uint8)
		assert(np.unique(img).shape[0] <= 2**bits)

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
		directories = sorted(os.listdir(datatype_dir))

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