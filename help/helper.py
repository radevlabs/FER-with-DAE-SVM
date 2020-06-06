from skimage.util.shape import view_as_blocks
import cv2
import numpy as np
import json
import sys
import os

def paths(base:str):
    base = base[:-1] if base[-1] == '/' else base
    return np.array([f'{base}/{path}' for path in os.listdir(base)])

def sprint(text):
    sys.stdout.write(text)

def printdict(dict, indent=0, keys_to_print=None):
    space = "  " * indent
    keys = (dict.keys() if keys_to_print is None else keys_to_print)
    rank_len = sorted(keys, key=lambda x: len(x), reverse=True)
    max_len = len(rank_len[0])
    for key in keys:
        if type(dict[key]) == type({}):
            print(space + str(key))
            printdict(dict[key], indent + 1)
        else:
            print(space + str(key) + (' ' * (max_len - len(key) + 1)) + ': ' + str(dict[key]))

def savejson(path, data):
    try:
        with open(path, 'w') as file:
            json.dump(data, file)
        return f'saved to {path}'
    except:
        return 'failed to save'
        
def loadjson(path):
    with open(path) as file:
        data = json.load(file)
    return data

def hasfile(path):
    return os.path.exists(path)

def aemodels(hfrom=1000, hto=600, hstep=100, cfrom=500, cto=50, cstep=50, n_layer=-1):
    layers = np.arange(hto, hfrom + 1, hstep)[::-1]
    clayers = np.arange(cto, cfrom + 1, cstep)[::-1]
    models = []
    for c in range(layers.shape[0] + 1):
        if n_layer == -1 or n_layer == c + 1:
            for clayer in clayers:
                model = []
                model.extend(layers[0:c])
                model.append(clayer)
                models.append(str(model).replace('[', '').replace(']', '').replace(', ', '-'))
    return np.array(models)

def dresize(img, size):
    x = img.shape[0]
    y = img.shape[1]
    scale = (size / x) if x < y else (size / y)
    return cv2.resize(img, (int(y * scale), int(x * scale)))

def swhere(word, data, reverse=False):
    arr = np.array([(word in d) for d in data])
    return arr == (not reverse)

def lbp(img):
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray_img = img.copy()
    resultimg = np.zeros((len(img), len(img[0])), np.uint8)
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            resultimg[i][j] = _hitung_pixel(gray_img, i, j)
    return resultimg

def _hitung_pixel(img, i, j):
    result = ''
    result += (_T(img, i - 1, j - 1, i, j)) # kiri atas
    result += (_T(img, i - 1, j, i, j))     # atas
    result += (_T(img, i - 1, j + 1, i, j)) # kanan atas
    result += (_T(img, i, j + 1, i, j))     # kanan
    result += (_T(img, i + 1, j + 1, i, j)) # kanan bawah
    result += (_T(img, i + 1, j, i, j))     # bawah
    result += (_T(img, i + 1, j - 1, i, j)) # kiri bawah
    result += (_T(img, i, j - 1, i, j))     # kiri
    return int(result, 2)

def _T(img, neighbor_i, neighbor_j, center_i, center_j):
    try:
        return '0' if img[neighbor_i][neighbor_j] >= img[center_i][center_j] else '1'
    except:
        return '1'

def face_detection(img, feature='haar', imscale=1, face_cascade=None, scaleFactor=1.3, minNeighbors=5):
    gray = cv2.resize(img, (int(img.shape[1] * imscale), int(img.shape[0] * imscale)))
    try:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    except:
        pass
    faces = []
    fc = face_cascade if face_cascade is not None else cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml' if feature == 'haar' else 'model/lbpcascade_frontalface.xml')
    faces_coords = fc.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for fx1, fy1, fw, fh in faces_coords:
        fx2 = fx1 + fw
        fy2 = fy1 + fh
        faces.append([fx1, fx2, fy1, fy2, fw, fh, fw * fh])
    faces = (np.array(faces) / imscale).astype(np.int)
    if len(faces) > 0:
        sorted_idx = np.argsort(-faces[:,6])
        faces = faces[sorted_idx]
    return faces

def crop_square(img, side=10, size=(128,128), imscale=1):
    dump = img.copy()
    size = np.array(size).astype(np.int)
    side = int(side)
    if side > 0:
        dump = cv2.resize(dump, (size[0] + (side * 2), size[1] + (side * 2)))
        dump = dump[side:-side,side:-side]
    else:
        dump = cv2.resize(dump, (size[0], size[1]))
    return dump

def hog_feature(img, cell_size=(8, 8), block_size=(2, 2), n_bins=9):
    # winSize is the size of the image cropped to an multiple of the cell size
    # cell_size is the size of the cells of the img patch over which to calculate the histograms
    # block_size is the number of cells which fit in the patch
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=n_bins)
    return hog.compute(img).flatten()

def blockviews(arr, size):
    arr = np.array(arr)
    y = np.ceil(arr.shape[0] / size[0]).astype(np.int)
    x = np.ceil(arr.shape[1] / size[1]).astype(np.int)
    views = []
    for i in range(y):
        view = []
        for j in range(x):
            view.append(arr[i * size[0]:(i + 1) * size[0], j * size[1]:(j + 1) * size[1]])
        views.append(view)
    return np.array(views)

def lbph(lbp_img, size=(16,16), normed=False):
    blocks = blockviews(lbp_img, size)
    histograms = []
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i,j].flatten()
            histogram, bins = np.histogram(block.flatten(), bins=256, range=(0, 256), normed=normed)
            histograms.append(histogram)
    histograms = np.array(histograms)
    return histograms.flatten()

def buildDAE(model_str, n_input, h_activation='relu', o_activation='sigmoid'):
    from keras.layers import Input, Dense
    from keras.models import Model
    
    emodel = model_str.split('-')
    emodel = list(map(int, emodel))
    
    coded_model = emodel[-1]
    del emodel[-1]
    
    dmodel = emodel.copy()
    dmodel.reverse()
    
    input_ = Input(shape=(n_input,))
    activations = []
    
    if len(emodel) > 0:
        encoded = Dense(emodel[0], activation=h_activation)(input_)
        activations.append(h_activation)
        del emodel[0]
        for n in emodel:
            encoded = Dense(n, activation=h_activation)(encoded)
            activations.append(h_activation)
        encoded = Dense(coded_model, activation=h_activation)(encoded)
        activations.append(h_activation)
    else:
        encoded = Dense(coded_model, activation=h_activation)(input_)
        activations.append(h_activation)
    
    if len(dmodel) > 0:
        decoded = Dense(dmodel[0], activation=h_activation)(encoded)
        activations.append(h_activation)
        del dmodel[0]
        for n in dmodel:
            decoded = Dense(n, activation=h_activation)(decoded)
            activations.append(h_activation)
        decoded = Dense(n_input, activation=o_activation)(decoded)
        activations.append(o_activation)
    else:
        decoded = Dense(n_input, activation=o_activation)(encoded)
        activations.append(o_activation)
    
    encoder = Model(input_, encoded)
    autoencoder = Model(input_, decoded)
    
    return autoencoder, encoder, model_str, activations

def undersampling(dataset, cols_target=-1, random_data=False, max_data=None):
    unique = np.unique(dataset[:,cols_target], return_counts=True)
    targets = unique[0]
    total = unique[1]
    targets = targets[total.argsort()]
    total.sort()
    n_min = total[0]
    
    if max_data is not None:
        max_data = int(max_data / targets.shape[0])
        n_min = max_data if n_min > max_data else n_min
    
    udataset = []
    for target in targets:
        data = dataset[dataset[:,cols_target] == target]
        if not random_data:
            udataset.extend(data[:n_min])
        else:
            idx = np.random.choice(np.arange(data.shape[0]), n_min, replace=False)
            udataset.extend(data[idx])
    
    return np.array(udataset)

def remove(data, item):
    try:
        data.remove(item)
    except:
        pass