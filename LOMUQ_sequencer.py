import numpy
#import threading
#import multiprocessing
#import ctypes
#from scipy import linalg
from scipy import ndimage
from keras.preprocessing.image import load_img, img_to_array, array_to_img#, save_img
# from keras.utils import load_img, img_to_array, array_to_img, save_img
from keras.utils import Sequence
import glob
#from PIL import Image
from skimage import util
from skimage import transform
import itertools
import os
import re
import h5py
import csv

# useZoom=False, zoom_factor_range=(0.95, 1.05), useAWGN=False, useMedian=False,
# useGaussian=False, useFlipping=False, cache=None, threadLockVar=None, useCache=False
#         self.useAWGN = useAWGN
#         self.useZoom = useZoom
#         self.zoom_factor_range = zoom_factor_range
#         self.useFlipping = useFlipping
#         self.useMedian = useMedian
#         self.medianSize = [0, 1, 3, 5, 7, 9, 11]
#         self.useGaussian = useGaussian
#         self.gaussianRange = (0, 0.075)

#         # ===============================#
#         # == caching-related variables ==#
#         # ===============================#
#         self.useCache = useCache
#         self.cache = cache
#         self._lock_ = threadLockVar
#         self.pid = 0
#         self.seeded = False
#         self._nsteps = int(numpy.ceil(len(self.fileArray) / float(self.batch_size)))

# locations in (lon, lat) format
sample_locations =[
    (75.0, 9.5),
    (78.0, 7.0),
    (80.5, 9.5),
    (80.5, 12.0),
    (80.5, 15.0),
    (81.5, 16.0),
    (83.0, 17.5),
    (85.0, 19.0),
    (86.5, 20.0),
    (87.25, 21.5),
    (90.0, 21.75),
    (92.0, 21.75),
    (93.0, 19.5),
    (04.0, 17.0),
    (94.0, 14.5),
    (94.5, 12.0),

    (96.0, 12.0),
    (98.0, 12.0),
    (100.0, 12.0),
    (102.5, 12.0),
    (105.0, 12.0),
    (107.5, 12.0),
    (110.0, 12.0),
    (112.5, 12.0),
    (115.0, 12.0),
    (117.5, 12.0),
    (120.0, 12.0),
    (122.5, 12.0),
    (125.0, 12.0),
    (127.5, 12.0),
    (130.0, 12.0),

    (96.0, 10.0),
    (98.0, 10.0),
    (100.0, 10.0),
    (102.5, 10.0),
    (105.0, 10.0),
    (107.5, 10.0),
    (110.0, 10.0),
    (112.5, 10.0),
    (115.0, 10.0),
    (117.5, 10.0),
    (120.0, 10.0),
    (122.5, 10.0),
    (125.0, 10.0),
    (127.5, 10.0),
    (130.0, 10.0),

    (96.0, 7.5),
    (98.0, 7.5),
    (100.0, 7.5),
    (102.5, 7.5),
    (105.0, 7.5),
    (107.5, 7.5),
    (110.0, 7.5),
    (112.5, 7.5),
    (115.0, 7.5),
    (117.5, 7.5),
    (120.0, 7.5),
    (122.5, 7.5),
    (125.0, 7.5),
    (127.5, 7.5),
    (130.0, 7.5),

    (96.0, 5.0),
    (98.0, 5.0),
    (100.0, 5.0),
    (102.5, 5.0),
    (105.0, 5.0),
    (107.5, 5.0),
    (110.0, 5.0),
    (112.5, 5.0),
    (115.0, 5.0),
    (117.5, 5.0),
    (120.0, 5.0),
    (122.5, 5.0),
    (125.0, 5.0),
    (127.5, 5.0),
    (130.0, 5.0),

    (96.0, 2.5),
    (98.0, 2.5),
    (100.0, 2.5),
    (102.5, 2.5),
    (105.0, 2.5),
    (107.5, 2.5),
    (110.0, 2.5),
    (112.5, 2.5),
    (115.0, 2.5),
    (117.5, 2.5),
    (120.0, 2.5),
    (122.5, 2.5),
    (125.0, 2.5),
    (127.5, 2.5),
    (130.0, 2.5),

    (96.0, 0.0),
    (98.0, 0.0),
    (100.0, 0.0),
    (102.5, 0.0),
    (105.0, 0.0),
    (107.5, 0.0),
    (110.0, 0.0),
    (112.5, 0.0),
    (115.0, 0.0),
    (117.5, 0.0),
    (120.0, 0.0),
    (122.5, 0.0),
    (125.0, 0.0),
    (127.5, 0.0),
    (130.0, 0.0),

    (96.0, -2.5),
    (98.0, -2.5),
    (100.0, -2.5),
    (102.5, -2.5),
    (105.0, -2.5),
    (107.5, -2.5),
    (110.0, -2.5),
    (112.5, -2.5),
    (115.0, -2.5),
    (117.5, -2.5),
    (120.0, -2.5),
    (122.5, -2.5),
    (125.0, -2.5),
    (127.5, -2.5),
    (130.0, -2.5),

    (96.0, -5.0),
    (98.0, -5.0),
    (100.0, -5.0),
    (102.5, -5.0),
    (105.0, -5.0),
    (107.5, -5.0),
    (110.0, -5.0),
    (112.5, -5.0),
    (115.0, -5.0),
    (117.5, -5.0),
    (120.0, -5.0),
    (122.5, -5.0),
    (125.0, -5.0),
    (127.5, -5.0),
    (130.0, -5.0),

    (96.0, -7.5),
    (98.0, -7.5),
    (100.0, -7.5),
    (102.5, -7.5),
    (105.0, -7.5),
    (107.5, -7.5),
    (110.0, -7.5),
    (112.5, -7.5),
    (115.0, -7.5),
    (117.5, -7.5),
    (120.0, -7.5),
    (122.5, -7.5),
    (125.0, -7.5),
    (127.5, -7.5),
    (130.0, -7.5),

    (96.0, -10.0),
    (98.0, -10.0),
    (100.0, -10.0),
    (102.5, -10.0),
    (105.0, -10.0),
    (107.5, -10.0),
    (110.0, -10.0),
    (112.5, -10.0),
    (115.0, -10.0),
    (117.5, -10.0),
    (120.0, -10.0),
    (122.5, -10.0),
    (125.0, -10.0),
    (127.5, -10.0),
    (130.0, -10.0),

    (96.0, -12.5),
    (98.0, -12.5),
    (100.0, -12.5),
    (102.5, -12.5),
    (105.0, -12.5),
    (107.5, -12.5),
    (110.0, -12.5),
    (112.5, -12.5),
    (115.0, -12.5),
    (117.5, -12.5),
    (120.0, -12.5),
    (122.5, -12.5),
    (125.0, -12.5),
    (127.5, -12.5),
    (130.0, -12.5),

    (125.0, -12.5),
    (127.5, -12.5),
    (130.0, -12.5),
    (132.5, -12.5),
    (135.0, -12.5),
    (137.5, -12.5),
    (140.0, -12.5),
    (142.5, -12.5),
    (145.0, -12.5),
    (145.0, -15.0),
    (146.5, -17.5),
    (150.0, -20.0),
    (152.5, -22.5),
    (155.0, -25.0),
    (155.0, -27.5),
    (155.0, -30.0),
    (155.0, -32.5),
    (152.5, -35.0),
    (150.0, -37.5),
    (147.5, -37.5),
    (145.0, -37.5),
    (142.5, -37.5),
    (140.0, -37.5),
    (137.5, -37.5),
    (150.0, -40.0),
    (147.5, -40.0),
    (145.0, -40.0),
    (142.5, -40.0),
    (150.0, -42.5),
    (147.5, -42.5),
    (145.0, -42.5),
    (142.5, -42.5),
    (150.0, -45.0),
    (147.5, -45.0),
    (145.0, -45.0),
    (142.5, -45.0),
    (135.0, -35.0),
    (132.5, -32.5),
    (130.0, -32.5),
    (127.5, -32.5),
    (125.0, -32.5),
    (122.5, -35.0),
    (120.0, -35.0),
    (117.5, -35.0),
    (115.0, -35.0),
    (115.0, -32.5),
    (115.0, -30.0),
    (112.5, -27.5),
    (112.5, -25.0),
    (112.5, -22.5),
    (112.5, -20.0),
    (115.0, -20.0),
    (117.5, -20.0),
    (120.0, -20.0),
    (122.5, -17.5),
    (125.0, -15.0),
    (122.5, -12.5),

    (165.0, -50.0),
    (167.5, -47.5),
    (170.0, -45.0),
    (172.5, -42.5),
    (175.0, -40.0),
    (177.5, -37.5),
    (180.0, -35.0),

    (-92.5, -2.5),
    (-90.0, -2.5),
    (-97.5, -2.5),
    (-92.5, 0.0),
    (-90.0, 0.0),
    (-97.5, 0.0),
    (-92.5, 2.5),
    (-90.0, 2.5),
    (-97.5, 2.5)
]

def tokenize(filename):
    digits = re.compile(r'(\d+)')
    return tuple(int(token) if match else token for token, match in
                 ((fragment, digits.search(fragment)) for fragment in digits.split(filename)))

def get_sample_location():
    index = numpy.random.randint(0, len(sample_locations), 1)
    return sample_locations[index]


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(numpy.round(h * zoom_factor))
        zw = int(numpy.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = numpy.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(numpy.round(h / zoom_factor))
        zw = int(numpy.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def numpy_normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


def get_min_max(a, numChannels, minx=None, maxx=None):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if numChannels<=1:
        if minx is None:
            minx = numpy.min(a)
        if maxx is None:
            maxx = numpy.max(a)
    else:
        if minx is None:
            minx = []
        if maxx is None:
            maxx = []
        if outDimLen < 4:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                if len(minx) < numChannels:
                    minx.append(numpy.min(a[:, :, channelIdx]))
                if len(maxx) < numChannels:
                    maxx.append(numpy.max(a[:, :, channelIdx]))
        else:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                if len(minx) < numChannels:
                    minx.append(numpy.min(a[:, :, :, channelIdx]))
                if len(maxx) < numChannels:
                    maxx.append(numpy.max(a[:, :, :, channelIdx]))
    #print(("{} in vs {} out vs {} a-shape".format(inShape,outShape, a.shape)))
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return minx, maxx


def normaliseFieldArray(a, numScenarios, numTimesteps):  # , minx=None, maxx=None
    minx = None
    maxx = None
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if numTimesteps <= 2:
        if minx is None:
            minx = numpy.min(a)
        if maxx is None:
            maxx = numpy.max(a)
        a = (a - minx) / (maxx-minx)
    else:
        if minx is None:
            minx = []
        if maxx is None:
            maxx = []
        if outDimLen < 4:
            minx.append(numpy.min(a))
            maxx.append(numpy.max(a))
            if numpy.fabs(maxx[-1] - minx[-1]) > 0:
                a = (a - minx[-1]) / (maxx[-1] - minx[-1])
        else:
            for trajectory_idx in itertools.islice(itertools.count(), 0, numScenarios):
                minx.append(numpy.min(a[trajectory_idx]))
                maxx.append(numpy.max(a[trajectory_idx]))
                if numpy.fabs(maxx[-1] - minx[-1]) > 0:
                    a[trajectory_idx] = (a[trajectory_idx] - minx[-1]) / (maxx[-1] - minx[-1])
    #print(("{} in vs {} out vs {} a-shape".format(inShape,outShape, a.shape)))
    if outDimLen != inDimLen:
        a = a.reshape(inShape)
    return minx, maxx, a


def denormaliseFieldArray(a, numChannels, minx=None, maxx=None):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if numChannels <= 1:
        a = a * (maxx - minx) + minx
    else:
        if outDimLen < 4:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                a[:, :, channelIdx] = a[:, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
        else:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                a[:, :, :, channelIdx] = a[:, :, :, channelIdx] * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return a


def clipNormFieldArray(a, numChannels):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    minx = None
    maxx = None

    if numChannels <= 1:
        minx = 0
        maxx = 200.0
        a = numpy.clip(a,minx,maxx)
    else:
        minx = numpy.zeros(32, a.dtype)
        # IRON-CAP
        #maxx = numpy.array([197.40414602,164.05027316,136.52565589,114.00212036,95.65716526,80.58945472,67.46342114,56.09140486,45.31409774,37.64459755,31.70887797,26.41859429,21.9482954,18.30031205,15.31461954,12.82080624,10.70525853,9.17048875,7.82142154,6.7137903,5.82180097,5.01058597,4.41808895,3.81359458,3.40606635,3.01021494,2.76689262,2.47842852,2.32304044,2.12137244,15.00464109,33.07879503], a.dtype)
        # SCANDIUM-CAP
        maxx = numpy.array([40.77704764,33.66334239,27.81001556,22.99326739,19.0324146,15.68578289,12.92738646,10.67188431,8.82938367,7.32395058,6.09176207,5.08723914,4.25534357,3.58350332,3.0290752,2.57537332,2.20811373,1.8954763,1.64371557,1.43238593,1.27925194,1.24131863,1.11358517,1.08348688,1.03346652,0.95083789,0.9814638,0.87886442,1.06108008,1.4744603,1.37953941,1.33551697])
        for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
            a[:, :, channelIdx] = numpy.clip(a[:, :, channelIdx], minx[channelIdx], maxx[channelIdx])
    if numChannels<=1:
        a = ((a - minx) / (maxx-minx)) * 2.0 - 1.0
    else:
        if outDimLen < 4:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                    a[:, :, channelIdx] = ((a[:, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
        else:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                if numpy.fabs(maxx[channelIdx] - minx[channelIdx]) > 0:
                    a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] - minx[channelIdx]) / (maxx[channelIdx] - minx[channelIdx])) * 2.0 - 1.0
    #print(("{} in vs {} out vs {} a-shape".format(inShape,outShape, a.shape)))
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return minx, maxx, a


def denormFieldArray(a, numChannels, minx=None, maxx=None):
    inShape = a.shape
    inDimLen = len(inShape)
    a = numpy.squeeze(a)
    outShape = a.shape
    outDimLen = len(outShape)
    if numChannels <= 1:
        a = ((a + 1.0) / 2.0) * (maxx - minx) + minx
    else:
        if outDimLen < 4:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                a[:, :, channelIdx] = ((a[:, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
        else:
            for channelIdx in itertools.islice(itertools.count(), 0, numChannels):
                a[:, :, :, channelIdx] = ((a[:, :, :, channelIdx] + 1.0) / 2.0) * (maxx[channelIdx] - minx[channelIdx]) + minx[channelIdx]
    if outDimLen<inDimLen:
        a = a.reshape(inShape)
    return a

class LOMUQ_sequencer(Sequence):

    def __init__(self, batch_size=1, image_size=(16, 16), target_size=(16, 16), input_timesteps=5, output_timesteps=1,
                 useResize=False, useCrop=False, useNormData=False, dir_path="", save_to_dir=None, save_format="png"):
        self.batch_size = batch_size
        self.image_size = image_size
        self.target_size = target_size
        self.in_timesteps = input_timesteps
        self.out_timesteps = output_timesteps
        self.uv_dtype_in = None
        self.pc_dtype_in = None
        self.y_dtype_in = None
        self.useResize = useResize
        self.useCrop = useCrop
        self.useNormData = useNormData
        self._epoch_num_ = 0
        self.numScenarios = 0

        self.useCache = False
        self.seeded = False
        self._nsteps = 1

        # ========================================#
        # == zoom-related image information ==#
        # ========================================#
        # self.im_center = None
        # self.im_shift = None
        # self.im_bounds = None
        # self.im_center = numpy.array([int(self.target_size[0] - 1) / 2, int(self.target_size[1] - 1) / 2], dtype=numpy.int32)
        # self.im_shift = numpy.array([int(self.image_size[0] - 1) / 2, int(self.image_size[1] - 1) / 2], dtype=numpy.int32)
        # left = max(self.im_shift[0] - self.im_center[0], 0)
        # right = min(left + self.target_size[0], self.image_size[0])
        # top = max(self.im_shift[1] - self.im_center[1], 0)
        # bottom = min(top + self.target_size[1], self.image_size[1])
        # self.im_bounds = (left, right, top, bottom)

        # ===================================#
        # == directory-related information ==#
        # ===================================#
        # self.info_file = []
        # self.grid_files = []
        # self.Ufiles = []
        # self.Vfiles = []
        # self.Lfiles = []
        # self.Dfiles = []
        # self.rDfiles = []
        # self.lrDfiles = []
        # self.PCfiles = []
        # self.particle_files = []
        self.Ufiles = []
        self.Vfiles = []
        self.PCfiles = []

        self.in_dir = dir_path
        self.save_to_dir = save_to_dir
        self.save_format = save_format
        self.store_img = True
        # ======================#
        # == batch size setup ==#
        # ======================#
        # self.batch_image_size_x = (self.batch_size, self.image_size[0], self.image_size[1], self.in_timesteps, 1)
        # self.batch_image_size_y = (self.batch_size, self.image_size[0], self.image_size[1], self.out_timesteps, 1)
        # if self.useCrop or self.useResize:
        #     self.batch_image_size_x = (
        #     self.batch_size, self.target_size[0], self.target_size[1], self.in_timesteps, 1)
        #     self.batch_image_size_y = (
        #     self.batch_size, self.target_size[0], self.target_size[1], self.out_timesteps, 1)
        self.batch_image_size_x = (self.batch_size, self.out_timesteps, self.target_size[0], self.target_size[1], 1)
        self.batch_image_size_y = (self.batch_size, self.target_size[0], self.target_size[1], 1)

    def prepareProcessedInput(self):
        self.Ufiles.append(str(self.in_dir) + "/hydrodynamic_U_dataList.h5")
        self.Vfiles.append(str(self.in_dir) + "/hydrodynamic_V_dataList.h5")
        self.PCfiles.append(str(self.in_dir) + "/particleCountList.h5")

        particleCountListFile = h5py.File(self.Ufiles[0], 'r')
        hydrodynamic_U_dataListFile = h5py.File(self.Vfiles[0], 'r')
        hydrodynamic_V_dataListFile = h5py.File(self.PCfiles[0], 'r')

        # particleCountList = particleCountListFile['ParticleCount'] # [()]
        # hydrodynamic_V_dataList = hydrodynamic_V_dataListFile['hydrodynamic_V'] # [()]
        # hydrodynamic_U_dataList = hydrodynamic_U_dataListFile['hydrodynamic_U'] # [()]

        if len(self.Ufiles) and os.path.exists(self.Ufiles[0]):
            f = h5py.File(self.Ufiles[0], 'r')
            imU = numpy.array(f['hydrodynamic_U'])
            f.close()
            self.uv_dtype_in = imU.dtype
            self.in_timesteps = imU.shape[1] if len(imU.shape) > 3 else imU.shape[0]
        if len(self.PCfiles) and os.path.exists(self.PCfiles[0]):
            f = h5py.File(self.PCfiles[0], 'r')
            imPC = numpy.array(f['ParticleCount'])
            f.close()
            self.pc_dtype_in = imPC.dtype
            self.numScenarios = imPC.shape[0]

        # ======================================== #
        # ==== crop-related image information ==== #
        # ======================================== #
        # self.im_center = None
        # self.im_shift = None
        # self.im_bounds = None
        # self.im_center = numpy.array([int(self.target_size[0] - 1) / 2, int(self.target_size[1] - 1) / 2], dtype=numpy.int32)
        # self.im_shift = numpy.array([(self.image_size[0] - 1) / 2, (self.image_size[1] - 1) / 2], dtype=numpy.int32)
        # left = max(self.im_shift[0] - self.im_center[0], 0)
        # right = min(left + self.target_size[0], self.image_size[0])
        # top = max(self.im_shift[1] - self.im_center[1], 0)
        # bottom = min(top + self.target_size[1], self.image_size[1])
        # self.im_bounds = (left, right, top, bottom)

    def prepareDirectFileInput(self, input_image_paths):
        i = 0
        for entry in input_image_paths:
            for name in glob.glob(os.path.join(entry, 'file.csv')):
                self.info_file.append(str(i)+"_"+name)
            for name in glob.glob(os.path.join(entry, 'grid.h5')):
                self.grid_files.append(str(i)+"_"+name)
            for name in glob.glob(os.path.join(entry, 'particles.h5')):
                self.particle_files.append(str(i)+"_"+name)
            for name in glob.glob(os.path.join(entry, '*U.h5')):
                self.Ufiles.append(str(i)+"_"+name)
            for name in glob.glob(os.path.join(entry, '*V.h5')):
                self.Vfiles.append(str(i)+"_"+name)
            for name in glob.glob(os.path.join(entry, 'density.h5')):
                self.Dfiles.append(str(i)+"_"+name)
            for name in glob.glob(os.path.join(entry, 'rel_density.h5')):
                self.rDfiles.append(str(i)+"_"+name)
                self.lrDfiles.append(str(i)+"_local"+"_"+name)
            for name in glob.glob(os.path.join(entry, 'lifetime.h5')):
                self.Lfiles.append(str(i)+"_"+name)
            for name in glob.glob(os.path.join(entry, 'particlecount.h5')):
                self.PCfiles.append(str(i)+"_"+name)
            i += 1
        # = Now you can sort your file names like so: =#
        self.info_file.sort(key=tokenize)
        self.grid_files.sort(key=tokenize)
        self.particle_files.sort(key=tokenize)
        self.Ufiles.sort(key=tokenize)
        self.Vfiles.sort(key=tokenize)
        self.Dfiles.sort(key=tokenize)
        self.rDfiles.sort(key=tokenize)
        self.Lfiles.sort(key=tokenize)
        self.PCfiles.sort(key=tokenize)
        self.numImages = len(self.particle_files)

        # if len(self.particle_files) and os.path.exists(self.particle_files[0]):
        #     f = h5py.File(self.Dfiles[0], 'r')
        #     imD = numpy.array(f['density'], order='F').transpose()
        #     f.close()
        #     self.y_dtype_in = imD.dtype
        #     if len(imD.shape) > 3:
        #         imD = numpy.squeeze(imD[:, :, 0, :])
        #     if len(imD.shape) < 3:
        #         imD = imD.reshape(imD.shape + (1,))
        #     if imD.shape != outImgDims:
        #         print("Error - read data D shape ({}) and expected data shape ({}) of X are not equal. EXITING ...".format(imD.shape, outImgDims))
        #         exit()

        center = (75.0, 9.5)
        if len(self.particle_files) and os.path.exists(self.particle_files[0]):
            siminfos = []
            with open(self.info_file[0], newline='') as csvfile:
                csvinfo = csv.reader(csvfile, delimiter=',')
                siminfos = csvinfo[1]
            gres = float(eval(siminfos[5]))
            xsteps, ysteps = int(gres * 360.0), int(gres * 180.0)
            f = h5py.File(self.particle_files[0], 'r')
            px = numpy.array(f['p_x'])
            py = numpy.array(f['p_y'])
            pt = numpy.array(f['p_t'])
            x_mask_min = px > center[0]-2.5
            x_mask_max = px < center[0]+2.5
            y_mask_min = py > center[1]-2.5
            y_mask_max = py < center[1]+2.5
            total_max = numpy.logical_and(numpy.logical_and(x_mask_min, x_mask_max), numpy.logical_and(y_mask_min, y_mask_max))
            pts_indices = numpy.nonzero(total_max)[0]
            n_pts_local = pts_indices.shape[0]
            pcounts = numpy.zeros((ysteps, xsteps), dtype=numpy.int32)
            A = float(gres**2)
            for ti in range(pt.shape[1]):
                x_in = numpy.array(px[:, ti])[pts_indices]
                y_in = numpy.array(py[:, ti])[pts_indices]
                xpts = numpy.floor((x_in+(360.0/2.0))*gres).astype(numpy.int32).flatten()
                ypts = numpy.floor((y_in+(180.0/2.0))*gres).astype(numpy.int32).flatten()
                for pi in range(xpts.shape[0]):
                    try:
                        pcounts[ypts[pi], xpts[pi]] += 1
                    except (IndexError, ) as error_msg:
                        # we don't abort here cause the brownian-motion wiggle of AvectionRK4EulerMarujama always edges on machine precision, which can np.floor(..) make go over-size
                        # print("\nError trying to index point ({}, {}) with indices ({}, {})".format(fX[pi, ti], fY[pi, ti], xpts[pi], ypts[pi]))
                        print("\nError trying to index point ({}, {}) ...".format(x_in[pi, ti], y_in[pi, ti]))
                        print("Point index: {}".format(pi))
                        print("Requested spatial index: ({}, {})".format(xpts[pi], ypts[pi]))
            density = pcounts.astype(numpy.float32) / A
            rel_density = density / float(n_pts_local)

            self.y_dtype_in = numpy.float32
            # imD = imD.reshape(imD.shape + (1,))

        if len(self.Ufiles) and os.path.exists(self.Ufiles[0]):
            self.in_dir = os.path.dirname(self.Ufiles[0])
            f = h5py.File(self.Ufiles[0], 'r')
            imU = numpy.array(f['uo'], order='F').transpose()
            f.close()
            self.x_dtype_in = imU.dtype
            # imU = imU.reshape(imU.shape + (1,))

        if len(self.Vfiles) and os.path.exists(self.Vfiles[0]):
            f = h5py.File(self.Vfiles[0], 'r')
            # define your variable names in here
            imV = numpy.array(f['vo'], order='F').transpose()
            f.close()
            # imV = imV.reshape(imV.shape + (1,))

        # ======================================== #
        # ==== crop-related image information ==== #
        # ======================================== #
        self.im_center = None
        self.im_shift = None
        self.im_bounds = None
        self.im_center = numpy.array([int(self.target_size[0] - 1) / 2, int(self.target_size[1] - 1) / 2],
                                     dtype=numpy.int32)
        self.im_shift = numpy.array([(self.image_size[0] - 1) / 2, (self.image_size[1] - 1) / 2], dtype=numpy.int32)
        left = max(self.im_shift[0] - self.im_center[0], 0)
        right = min(left + self.target_size[0], self.image_size[0])
        top = max(self.im_shift[1] - self.im_center[1], 0)
        bottom = min(top + self.target_size[1], self.image_size[1])
        self.im_bounds = (left, right, top, bottom)

    def _initCache_locked_(self):
        loadData_flag = True
        while (loadData_flag):
            ii = 0
            with self._lock_:
                loadData_flag = (self.cache.is_cache_updated() == False)
                ii = self.cache.get_renew_index()
            if loadData_flag == False:
                break
            file_index = numpy.random.randint(0, self.numImages)
            inName = self.fileArray[file_index]
            f = h5py.File(inName, 'r')
            imX = numpy.array(f['Data_X'], order='F').transpose()
            imY = numpy.array(f['Data_Y'], order='F').transpose()
            f.close()
            if len(imX.shape) != len(imY.shape):
                print("Image dimensions do not match - EXITING ...")
                exit(1)

            if len(imX.shape) > 3:
                indices = [ii, ]
                while (len(indices) < imX.shape[2]) and (loadData_flag == True):
                    with self._lock_:
                        ii = self.cache.get_renew_index()
                        indices.append(ii)
                    with self._lock_:
                        loadData_flag = (self.cache.is_cache_updated() == False)
                slice_indices = numpy.random.randint(0, imX.shape[2], len(indices))
                """
                Data Normalisation
                """
                minValX = None
                maxValX = None
                minValY = None
                maxValY = None
                if self.useNormData:
                    """
                    Data Normalisation
                    """
                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels)
                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, minx=minValX, maxx=maxValX)

                for index in itertools.islice(itertools.count(), 0, len(indices)):
                    slice_index = slice_indices[index]
                    imX_slice = numpy.squeeze(imX[:, :, slice_index, :])
                    # === we need to feed the data as 3D+1 channel data stack === #
                    if len(imX_slice.shape) < 3:
                        imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                    if len(imX_slice.shape) < 4:
                        imX_slice = imX_slice.reshape(imX_slice.shape + (1,))
                    imY_slice = numpy.squeeze(imY[:, :, slice_index, :])
                    if len(imY_slice.shape) < 3:
                        imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                    if len(imY_slice.shape) < 4:
                        imY_slice = imY_slice.reshape(imY_slice.shape + (1,))
                    # == Note: do data normalization here to reduce memory footprint ==#
                    imX_slice = imX_slice.astype(numpy.float32)
                    imY_slice = imY_slice.astype(numpy.float32)
                    with self._lock_:
                        self.cache.set_cache_item_x(indices[index], imX_slice)
                        self.cache.set_item_limits_x(indices[index], minValX, maxValX)
                        self.cache.set_cache_item_y(indices[index], imY_slice)
                        self.cache.set_item_limits_y(indices[index], minValY, maxValY)

            else:
                # === we need to feed the data as 3D+1 channel data stack === #
                if len(imX.shape) < 3:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imX.shape) < 4:
                    imX = imX.reshape(imX.shape + (1,))
                if len(imY.shape) < 3:
                    imY = imY.reshape(imY.shape + (1,))
                if len(imY.shape) < 4:
                    imY = imY.reshape(imY.shape + (1,))
                # == Note: do data normalization here to reduce memory footprint ==#
                """
                Data Normalisation
                """
                minValX = None
                maxValX = None
                minValY = None
                maxValY = None
                if self.useNormData:
                    minValX, maxValX, imX = normaliseFieldArray(imX, self.input_channels)
                    minValY, maxValY, imY = normaliseFieldArray(imY, self.output_channels, minx=minValX, maxx=maxValX)
                imX = imX.astype(numpy.float32)
                imY = imY.astype(numpy.float32)
                with self._lock_:
                    self.cache.set_cache_item_x(ii, imX)
                    self.cache.set_item_limits_x(ii, minValX, maxValX)
                    self.cache.set_cache_item_y(ii, imY)
                    self.cache.set_item_limits_y(ii, minValY, maxValY)

            with self._lock_:
                loadData_flag = (self.cache.is_cache_updated() == False)
        return

    def set_nsteps(self, nsteps):
        self._nsteps = nsteps

    def __len__(self):
        return self._nsteps

    #    self._nsteps = int(numpy.ceil(len(self.fileArray)/float(self.batch_size)))
    #    return int(numpy.ceil(len(self.fileArray)/float(self.batch_size)))

    def __getitem__(self, idx):
        self.pid = os.getpid()
        if self.seeded == False:
            numpy.random.seed(self.pid)
            self.seeded = True

        if self.useCache:
            pass
        #     flushCache = False
        #     with self._lock_:
        #         flushCache = (self.cache.is_cache_updated() == False)
        #     if flushCache == True:
        #         self._initCache_locked_()

        batchU = numpy.zeros(self.batch_image_size_x, dtype=self.uv_dtype_in)
        batchV = numpy.zeros(self.batch_image_size_x, dtype=self.uv_dtype_in)
        batchPC = numpy.zeros(self.batch_image_size_x, dtype=self.pc_dtype_in)
        batchY = numpy.zeros(self.batch_image_size_y, dtype=self.y_dtype_in)
        idxArray = None
        dataU = None
        minValU = None
        maxValU = None
        dataV = None
        minValV = None
        maxValV = None
        dataPC = None
        minValPC = None
        maxValPC = None
        if self.useCache:
            idxArray = numpy.random.randint(0, self.cache.get_cache_size(), self.batch_size)
        else:
            # imgIndex = min([(idx*self.batch_size)+j, self.numImages-1,len(self.fileArray)-1])
            # imgIndex = ((idx * self.batch_size) + j) % (self.numTrajectories - 1)
            """
            Load data from disk
            """
            fu = h5py.File(self.Ufiles[0], 'r')
            dataU = numpy.array(fu['hydrodynamic_U'])
            fu.close()
            fv = h5py.File(self.Vfiles[0], 'r')
            dataV = numpy.array(fv['hydrodynamic_V'])
            fv.close()
            fpcount = h5py.File(self.PCfiles[0], 'r')
            dataPC = numpy.array(fpcount['ParticleCount'])
            fpcount.close()

            if len(dataU.shape) < 3:
                dataU = numpy.array([dataU, ]*4, dataU.dtype)
            if len(dataU.shape) < 4:
                dataU = numpy.expand_dims(dataU, axis=0)
            if len(dataV.shape) < 3:
                dataV = numpy.array([dataV, ]*4, dataV.dtype)
            if len(dataV.shape) < 4:
                dataV = numpy.expand_dims(dataV, axis=0)
            if len(dataPC.shape) < 3:
                dataPC = numpy.array([dataPC, ]*self.numScenarios, dataPC.dtype)
            if len(dataPC.shape) < 4:
                dataPC = numpy.expand_dims(dataPC, axis=0)

            if dataU.shape != dataPC.shape:
                raise RuntimeError("Input- and Output sizes do not match.")

        for j in itertools.islice(itertools.count(), 0, self.batch_size):
            # sample_index = ((idx * self.batch_size) + j)
            sampleIndex = numpy.random.randint(0, self.numScenarios)
            # if self.useCache:
            #     pass
            #     # imgIndex = numpy.random.randint(0, self.cache_size)
            #     imgIndex = idxArray[j]
            #     with self._lock_:
            #         imX = self.cache.get_cache_item_x(imgIndex)
            #         minValX, maxValX = self.cache.get_item_limits_x(imgIndex)
            #         imY = self.cache.get_cache_item_y(imgIndex)
            #         minValY, maxValY = self.cache.get_item_limits_y(imgIndex)
            # else:
            #     # imgIndex = min([(idx*self.batch_size)+j, self.numImages-1,len(self.fileArray)-1])
            #     # imgIndex = ((idx * self.batch_size) + j) % (self.numTrajectories - 1)
            #     """
            #     Load data from disk
            #     """
            #     fu = h5py.File(self.Ufiles[0], 'r')
            #     dataU = numpy.array(fu['hydrodynamic_U'])
            #     fu.close()
            #     fv = h5py.File(self.Vfiles[0], 'r')
            #     dataV = numpy.array(fv['hydrodynamic_V'])
            #     fv.close()
            #     fpcount = h5py.File(self.PCfiles[0], 'r')
            #     dataPC = numpy.array(fpcount['ParticleCount'])
            #     fpcount.close()
            #
            #     if len(dataU.shape) < 3:
            #         dataU = numpy.array([dataU, ]*4, dataU.dtype)
            #     if len(dataU.shape) < 4:
            #         dataU = numpy.expand_dims(dataU, axis=0)
            #     if len(dataV.shape) < 3:
            #         dataV = numpy.array([dataV, ]*4, dataV.dtype)
            #     if len(dataV.shape) < 4:
            #         dataV = numpy.expand_dims(dataV, axis=0)
            #     if len(dataPC.shape) < 3:
            #         dataPC = numpy.array([dataPC, ]*4, dataPC.dtype)
            #     if len(dataPC.shape) < 4:
            #         dataPC = numpy.expand_dims(dataPC, axis=0)
            #
            #     if dataU.shape != dataPC.shape:
            #         raise RuntimeError("Input- and Output sizes do not match.")
            fname_in = "img_{}_{}_{}_{}".format(self._epoch_num_, self.pid, idx, j)

            """
            Data transform
            """
            valid_sample = False
            timeIndex = -1
            rowIndex = -1
            colIndex = -1
            while not valid_sample:
                valid_sample = True
                timeIndex = numpy.random.randint(0, (self.in_timesteps - self.out_timesteps)-1)
                sum_particles = numpy.sum(numpy.squeeze(dataPC[sampleIndex])[timeIndex:timeIndex + self.out_timesteps], axis=0)
                maskpc = numpy.nonzero(sum_particles)
                min_row_pc = numpy.min(maskpc[0])
                max_row_pc = numpy.max(maskpc[0])
                min_col_pc = numpy.min(maskpc[1])
                max_col_pc = numpy.max(maskpc[1])
                if (max_row_pc - min_row_pc) <= self.target_size[0]:
                    print("( max occupied row = {} - min occupied row = {} ) >  {} ?".format(max_row_pc, min_row_pc, self.target_size[0]))
                    valid_sample = False
                if (max_col_pc - min_col_pc) <= self.target_size[1]:
                    print("( max occupied column = {} - min occupied column = {} ) >  {} ?".format(max_col_pc, min_col_pc, self.target_size[1]))
                    valid_sample = False
                if valid_sample:
                    rowIndex = numpy.random.randint(0, max_row_pc - self.target_size[0])
                    colIndex = numpy.random.randint(0, max_col_pc - self.target_size[1])

            sample_U = numpy.array(dataU[sampleIndex, timeIndex:timeIndex + self.out_timesteps, rowIndex:rowIndex + self.target_size[0], colIndex:colIndex + self.target_size[1]]).squeeze()
            sample_V = numpy.array(dataV[sampleIndex, timeIndex:timeIndex + self.out_timesteps, rowIndex:rowIndex + self.target_size[0], colIndex:colIndex + self.target_size[1]]).squeeze()
            sample_PC = numpy.array(dataPC[sampleIndex, timeIndex:timeIndex + self.out_timesteps, rowIndex:rowIndex + self.target_size[0], colIndex:colIndex + self.target_size[1]]).squeeze()
            result_PC = numpy.array(dataPC[sampleIndex, timeIndex + self.out_timesteps, rowIndex:rowIndex + self.target_size[0], colIndex:colIndex + self.target_size[1]]).squeeze()


            # input_target_size = self.target_size + (self.input_channels,)
            # output_target_size = self.target_size + (self.output_channels,)

            # == Note: do data normalization here to reduce memory footprint ==#
            """
            Data Normalisation
            """
            if self.useNormData:
                minValU, maxValU, dataU = normaliseFieldArray(sample_U, self.numScenarios, self.in_timesteps)
                minValV, maxValV, dataV = normaliseFieldArray(sample_V, self.numScenarios, self.in_timesteps)
                minValPC, minValPC, dataPC = normaliseFieldArray(sample_PC, self.numScenarios, self.in_timesteps)
            sample_U = sample_U.astype(numpy.float32)
            sample_V = sample_V.astype(numpy.float32)
            sample_U = numpy.expand_dims(sample_U, axis=-1)
            sample_V = numpy.expand_dims(sample_V, axis=-1)
            sample_PC = numpy.expand_dims(sample_PC, axis=-1)
            result_PC = numpy.expand_dims(result_PC, axis=-1)

            """
            Store data if requested
            """
            if (self.save_to_dir is not None) and (self.store_img == True):
                # print("Range phantom (after scaling): {}; scale: {}; shape {}".format([numpy.min(imX), numpy.max(imX)], [min(minValX), max(maxValX)], imX.shape))
                # print("Range FBP (after scaling): {}; scale: {}; shape {}".format([numpy.min(imY), numpy.max(imY)], [min(minValY), max(maxValY)], imY.shape))
                # store_imu = sample_U[:, :, 0, 0]
                store_imu = sample_U[0, :, :, 0]
                if len(store_imu.shape) < 3:
                    store_imu = store_imu.reshape(store_imu.shape + (1,))
                sUImg = array_to_img(store_imu, data_format='channels_last')
                # save_img(os.path.join(self.save_to_dir,fname_in+"."+self.save_format),sUImg)
                sUImg.save(os.path.join(self.save_to_dir, fname_in + "_inputU." + self.save_format))
                store_imv = sample_V[0, :, :, 0]
                if len(store_imv.shape) < 3:
                    store_imv = store_imv.reshape(store_imv.shape + (1,))
                sVImg = array_to_img(store_imu, data_format='channels_last')
                # save_img(os.path.join(self.save_to_dir,fname_in+"."+self.save_format),sVImg)
                sVImg.save(os.path.join(self.save_to_dir, fname_in + "_inputV." + self.save_format))

                store_impc = sample_PC[0, :, :, 0]
                if len(store_impc.shape) < 3:
                    store_impc = store_impc.reshape(store_impc.shape + (1,))
                sPCImg = array_to_img(store_impc, data_format='channels_last')
                # save_img(os.path.join(self.save_to_dir,fname_out+"."+self.save_format), sPCImg)
                sPCImg.save(os.path.join(self.save_to_dir, fname_in + "_outputPC." + self.save_format))
            batchU[j] = sample_U
            batchV[j] = sample_V
            batchPC[j] = sample_PC
            batchY[j] = result_PC

        # === Comment Chris: only store images on the first epoch - not on all === #
        self.store_img = False
        return [batchU, batchV, batchPC], batchY

    def on_epoch_end(self):
        self._epoch_num_ = self._epoch_num_ + 1
        # print("Epoch: {}, num. CT gens called: {}".format(self._epoch_num_, self.fan_beam_CT.getNumberOfTransformedData()))
