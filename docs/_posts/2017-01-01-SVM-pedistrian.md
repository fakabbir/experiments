---
title: "Detecting Pedestrians with Support Vector Machines"
---

```python
data_location = r'\data'
```

```python
datadir = "data"
dataset = "\pedestrians128x64"
datafile = "%s%s.tar.gz" % (data_location, dataset)
```

```python
extractdir = "%s%s" % (data_location, dataset)
```

```python
import os
def extract_tar(filename):
    try:
        import tarfile
    except ImportError:
        raise ImportError("You do not have tarfile installed. "
                          "Try unzipping the file outside of "
                          "Python.")
    if True : #dataset[1:] not in os.listdir(extractdir):
        tar = tarfile.open(datafile)
        tar.extractall(path=extractdir)
        tar.close()
        print("{} sucessfully extracted to {}".format(datafile,
                                                      extractdir))
    else:
        print("Already Extracted")
```

```python
extract_tar(datafile)
```

    Already Extracted

```python
import cv2
```

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
os.listdir(extractdir)
```

    ['pedestrians128x64']

```python
for i in range(5):
    filename = "%s\pedestrians128x64\per0010%d.ppm" % (extractdir, i)
    #print(filename)
    img = cv2.imread(filename)
    plt.subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
```

![png](_images/output_12_0.png)

```python
win_size = (48, 96) # Most important argument for HOG
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                        cell_size, num_bins)
```

```python
import numpy as np
import random
random.seed(42)
X_pos = []
for i in random.sample(range(900), 400):
    filename = "%s\pedestrians128x64\per%05d.ppm" % (extractdir, i)
    img = cv2.imread(filename)
    if img is None:
        print('Could not find image')
        continue
    X_pos.append(hog.compute(img, (64, 64)))
```

    Could not find image

```python
len(X_pos)
```

    399

```python
X_pos = np.array(X_pos, dtype=np.float32)
y_pos = np.ones(X_pos.shape[0], dtype=np.int32)
X_pos.shape, y_pos.shape
```

    ((399, 1980, 1), (399,))

```python
negdir = r'\data\pedestrians_neg\pedestrians_neg'
hroi = 128
wroi = 64
X_neg = []
for negfile in os.listdir(negdir):
    filename = '%s\%s' % (negdir, negfile)
    #print(filename)
    img = cv2.imread(filename)
    img = cv2.resize(img,(512,512))
    for j in range(5):
        rand_y = random.randint(0, img.shape[0] - hroi)
        rand_x = random.randint(0, img.shape[1] - wroi)
        roi = img[rand_y:rand_y + hroi, rand_x:rand_x + wroi, :]
        X_neg.append(hog.compute(roi, (64, 64)))
```

```python
X_neg = np.array(X_neg, dtype=np.float32)
y_neg = -np.ones(X_neg.shape[0], dtype=np.int32)
X_neg.shape, y_neg.shape
```

    ((250, 1980, 1), (250,))

```python
X = np.concatenate((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))
```

```python
from sklearn import model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
def train_svm(X_train, y_train):
    svm = cv2.ml.SVM_create()
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm
```

```python
def score_svm(svm, X, y):
    from sklearn import metrics
    _, y_pred = svm.predict(X)
    return metrics.accuracy_score(y, y_pred)
```

```python
svm = train_svm(X_train, y_train)
```

```python
score_svm(svm, X_train, y_train)
```

    1.0

```python
score_svm(svm, X_test, y_test)
```

    0.6461538461538462

```python
score_train = []
score_test = []
for j in range(3):
    svm = train_svm(X_train, y_train)
    score_train.append(score_svm(svm, X_train, y_train))
    score_test.append(score_svm(svm, X_test, y_test))
    _, y_pred = svm.predict(X_test)
    false_pos = np.logical_and((y_test.ravel() == -1),
                               (y_pred.ravel() == 1))
    if not np.any(false_pos):
        print('no more false positives: done')
        break

    X_train = np.concatenate((X_train,
                              X_test[false_pos, :]),
                             axis=0)

    y_train = np.concatenate((y_train, y_test[false_pos]),axis=0)
```

    no more false positives: done

```python
score_train
```

    [1.0, 1.0]

```python
score_test
```

    [0.6461538461538462, 1.0]

```python
img_test = cv2.imread(r'C:\Users\fakabbir.amin\Desktop\DayScholar\OpenCV\opencv-machine-learning\notebooks\data\chapter6\pedestrian_test.JPG')
```

```python
stride = 16
found = []
for ystart in np.arange(0, img_test.shape[0], stride):
    for xstart in np.arange(0, img_test.shape[1], stride):
        if ystart + hroi > img_test.shape[0]:
            continue
        if xstart + wroi > img_test.shape[1]:
            continue
        roi = img_test[ystart:ystart + hroi,
                       xstart:xstart + wroi, :]
        feat = np.array([hog.compute(roi, (64, 64))])
        _, ypred = svm.predict(feat)
        if np.allclose(ypred, 1):
            found.append((ystart, xstart, hroi, wroi))
```

```python
rho, _, _ = svm.getDecisionFunction(0)
sv = svm.getSupportVectors()
hog.setSVMDetector(np.append(sv.ravel(), rho))
```

    ---------------------------------------------------------------------------

    error                                     Traceback (most recent call last)

    <ipython-input-70-445de3dc92d7> in <module>()
          1 rho, _, _ = svm.getDecisionFunction(0)
          2 sv = svm.getSupportVectors()
    ----> 3 hog.setSVMDetector(np.append(sv.ravel(), rho))


    error: OpenCV(3.4.4) C:\projects\opencv-python\opencv\modules\objdetect\src\hog.cpp:117: error: (-215:Assertion failed) checkDetectorSize() in function 'cv::HOGDescriptor::setSVMDetector'

```python
found = hog.detectMultiScale(img_test)
```

```python
hogdef = cv2.HOGDescriptor()
pdetect = cv2.HOGDescriptor_getDefaultPeopleDetector()
hogdef.setSVMDetector(pdetect)
found, _ = hogdef.detectMultiScale(img_test)
```

```python
from matplotlib import patches
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
```

    <matplotlib.image.AxesImage at 0x230eb43c080>

![png](_images/output_37_1.png)

```python
for f in found:
    ax.add_patch(patches.Rectangle((f[0], f[1]), f[2], f[3],
                                   color='y', linewidth=3,
                                   fill=False))
ax.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
```

    <matplotlib.image.AxesImage at 0x230eb54ae80>

```python
len(found)
```

    3

```python

```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-81-659a91e1816a> in <module>()
    ----> 1 plt.plot(ax)


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\pyplot.py in plot(*args, **kwargs)
       3356                       mplDeprecation)
       3357     try:
    -> 3358         ret = ax.plot(*args, **kwargs)
       3359     finally:
       3360         ax._hold = washold


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\__init__.py in inner(ax, *args, **kwargs)
       1853                         "the Matplotlib list!)" % (label_namer, func.__name__),
       1854                         RuntimeWarning, stacklevel=2)
    -> 1855             return func(ax, *args, **kwargs)
       1856
       1857         inner.__doc__ = _add_data_doc(inner.__doc__,


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\axes\_axes.py in plot(self, *args, **kwargs)
       1526
       1527         for line in self._get_lines(*args, **kwargs):
    -> 1528             self.add_line(line)
       1529             lines.append(line)
       1530


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\axes\_base.py in add_line(self, line)
       1930             line.set_clip_path(self.patch)
       1931
    -> 1932         self._update_line_limits(line)
       1933         if not line.get_label():
       1934             line.set_label('_line%d' % len(self.lines))


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\axes\_base.py in _update_line_limits(self, line)
       1952         Figures out the data limit of the given line, updating self.dataLim.
       1953         """
    -> 1954         path = line.get_path()
       1955         if path.vertices.size == 0:
       1956             return


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\lines.py in get_path(self)
        949         """
        950         if self._invalidy or self._invalidx:
    --> 951             self.recache()
        952         return self._path
        953


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\lines.py in recache(self, always)
        655         if always or self._invalidy:
        656             yconv = self.convert_yunits(self._yorig)
    --> 657             y = _to_unmasked_float_array(yconv).ravel()
        658         else:
        659             y = self._y


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\matplotlib\cbook\__init__.py in _to_unmasked_float_array(x)
       2048         return np.ma.asarray(x, float).filled(np.nan)
       2049     else:
    -> 2050         return np.asarray(x, float)
       2051
       2052


    c:\users\fakabbir.amin\desktop\cgi\tf_venv\lib\site-packages\numpy\core\numeric.py in asarray(a, dtype, order)
        499
        500     """
    --> 501     return array(a, dtype, copy=False, order=order)
        502
        503


    TypeError: float() argument must be a string or a number, not 'AxesSubplot'

![png](_images/output_40_1.png)
