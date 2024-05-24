import json
import site
from pathlib import Path

import cv2
import numpy as np

from siamese_network import SiameseNetwork


for sp in site.getsitepackages():
  if "site-packages" in sp:
    cv2.samples.addSamplesDataSearchPath(sp+"\\cv2\\data")

class Siamese:
  def __init__(self, weight_path,):
    seed = 0
    width = 105
    height = 105
    cells = 1
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    optimizer = 'adam'
    dropout_rate = 0.1
    self.net = SiameseNetwork(seed, width, height, cells, loss, metrics, optimizer, dropout_rate)
    self.set_weight(weight_path)

  def set_weight(self, weight_path):
    self.net._load_weights(str(weight_path))

  def preprocess(self, image, norm=False):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (105, 105))
    if norm:
      img = img/255
    return img.reshape(1, 105, 105, 1).astype(np.float64)

  def predict(self, imgs, norm=False):
    img1, img2 = imgs
    inp1 = self.preprocess(img1, norm)
    inp2 = self.preprocess(img2, norm)
    res = self.net.siamese_net.predict([inp1, inp2])
    return res[0]


class ImageLoader:
  def __init__(self, pair_list_file, img_path, crop_face=False):
    self.img_path = img_path
    self.crop_face = crop_face
    self.haar = None
    if self.crop_face:
      self.haar = cv2.CascadeClassifier(cv2.samples.findFile("haarcascade_frontalface_alt.xml"))

    with open(pair_list_file) as f:
      lines = f.readlines()

    self.nb_pairs = len(lines)
    self.pairs = []
    for line in lines:
      line = line.split()
      if len(line) == 4:
        cls = 0
        p1, a, p2, b = line
      elif len(line) == 3:
        cls = 1
        p1, a, b = line
        p2 = None
      p2 = p2 or p1
      self.pairs.append([[p1, a, p2, b], cls])

  def __iter__(self):
    self.count = 0
    return self

  def __len__(self):
    return self.nb_pairs

  def __next__(self):
    if self.count == self.nb_pairs:
      raise StopIteration

    pair, cls = self.pairs[self.count]
    path1, path2 = self._get_pair_path(pair)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if self.crop_face:
      img1 = self._crop_face(img1)
      img2 = self._crop_face(img2)
    self.count += 1
    return " ".join(pair), [img1, img2], cls

  def _crop_face(self, img):
    faces = self.haar.detectMultiScale(img, 1.2, 5)
    x, y, w, h = faces[0]
    return img[y:y + h, x:x + w]

  def _get_pair_path(self, pair):
    p1, a, p2, b = pair
    path1 = str((self.img_path/p1/(f"{p1}_{int(a):04d}.jpg")))
    path2 = str((self.img_path/p2/(f"{p2}_{int(b):04d}.jpg")))
    return path1, path2


if __name__ == "__main__":
  import time

  DATA_DIR = Path("lfw2/lfw2")
  WEIGHT_DIR = Path("weights/trained.h5")
  TEST_DESC_FILE = Path("lfw2/splits/test.txt")
  RESULTS_DIR = Path("results")

  net = Siamese(WEIGHT_DIR)
  loader = ImageLoader(TEST_DESC_FILE, DATA_DIR, crop_face=True)

  results = []
  for line, x, y in loader:
    r = net.predict(x)
    results.append({
      "pairs": line,
      "actual": y,
      "predicted": int(r[0] >= 0.5),
      "predicted_raw": float(r[0])
    })

  with (RESULTS_DIR / f"res_{int(time.time())}.json").open("w") as f:
    json.dump(results, f, indent=2)
