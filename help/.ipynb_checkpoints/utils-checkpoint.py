from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern, hog
from keras.utils import Sequence
import cv2
import time
import numpy as np

class DatasetSequence(Sequence):
    def __init__(self, path, batch_size, parallel=True, backend='loky'):
        self.path = path
        self.batch_size = batch_size
        self.parallel = parallel
        self.backend = backend

    def __len__(self):
        return int(np.ceil(len(self.path) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_path = self.path[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.parallel:
            batch_x = np.array(Parallel(n_jobs=-1, backend=self.backend)(delayed(self.__vector)(path) for path in batch_path))
        else:
            batch_x = np.array([self.__vector(path) for path in batch_path])
        return batch_x, batch_x
    
    def __vector(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        vector = []
        vector.extend(hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
        vector.extend(helper.lbph(local_binary_pattern(img, P=8, R=1), size=(16, 16), normed=True))
        return vector

class Timer:
    def __init__(self):
        self.years = 0
        self.months = 0
        self.days = 0
        self.hours = 0
        self.minutes = 0
        self.seconds = 0.
    
    def start(self):
        self.__start = time.time()
        return self
    
    def end(self):
        self.seconds = time.time() - self.__start
        self.seconds = float(self.seconds)
        self.minutes = self.seconds // 60
        self.hours = self.minutes // 60
        self.days = self.hours // 24
        self.months = self.days // 30
        self.years = self.months // 12
        
        self.seconds -= (self.minutes * 60)
        self.minutes -= (self.hours * 60)
        self.hours -= (self.days * 24)
        self.days -= (self.months * 30)
        self.months -= (self.years * 12)
        
        self.years = int(self.years)
        self.months = int(self.months)
        self.days = int(self.days)
        self.hours = int(self.hours)
        self.minutes = int(self.minutes)
        return self
    
    def summary(self, f=2, comma='.'):
        seconds = np.around(self.seconds, f) if f > 0 else int(self.seconds)
        t = [self.years, self.months, self.days, self.hours, self.minutes, seconds]
        s = ['tahun', 'bulan', 'hari', 'jam', 'menit', 'detik']
        smmr = ''
        for c in range(len(t)):
            if t[c] != 0:
                smmr += f'{t[c]} {s[c]} '
        return smmr[:-1].replace('.', comma)