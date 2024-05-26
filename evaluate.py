import argparse
import site
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from metrics import *
from plot import plot_curve
from siamese_network import SiameseNetwork

for sp in site.getsitepackages():
  if "site-packages" in sp:
    cv2.samples.addSamplesDataSearchPath(sp+"\\cv2\\data")

METRICS = ["Precision", "Recall", "Specificity", "F1-score", "F2-score", "F.5-score", "Accuracy"]

class Siamese:
  def __init__(self, weight_path=None):
    seed = 0
    width = 105
    height = 105
    cells = 1
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    optimizer = 'adam'
    dropout_rate = 0.1
    self.net = SiameseNetwork(seed, width, height, cells, loss, metrics, optimizer, dropout_rate)
    if weight_path is not None:
      self.set_weight(weight_path)

  def set_weight(self, weight_path):
    self.net._load_weights(str(weight_path))

  def preprocess(self, imgs):
    res = []
    for image in imgs:
      img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, (105, 105))
      res.append(img.reshape(105, 105, 1).astype(np.float64))
    return np.array(res)

  def predict(self, imgs, batch_size=1, preprocess=False):
    if preprocess:
      imgs = [self.preprocess(img) for img in imgs]
    res = self.net.siamese_net.predict(imgs, batch_size=batch_size)
    return res[:, 0]


class ImageLoader:
  def __init__(self, pair_list_file, img_path, preprocess=True, crop_face=False):
    self.img_path = img_path
    self.preproc = preprocess
    self.crop_face = crop_face
    self.haar = None
    if self.crop_face:
      self.haar = cv2.CascadeClassifier(cv2.samples.findFile("haarcascade_frontalface_alt.xml"))

    with open(pair_list_file, 'r') as f:
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
    imgs = [cv2.imread(path) for path in self._get_pair_path(pair)]
    if self.crop_face:
      imgs = [self._crop_face(img) for img in imgs]
    if self.preproc:
      imgs = [self._preprocess(img) for img in imgs]

    self.count += 1
    return " ".join(pair), imgs, cls

  def _crop_face(self, img):
    faces = self.haar.detectMultiScale(img, 1.2, 5)
    x, y, w, h = faces[0]
    return img[y:y + h, x:x + w]

  def _get_pair_path(self, pair):
    p1, a, p2, b = pair
    path1 = str((self.img_path/p1/(f"{p1}_{int(a):04d}.jpg")))
    path2 = str((self.img_path/p2/(f"{p2}_{int(b):04d}.jpg")))
    return path1, path2

  def _preprocess(self, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (105, 105))
    return img.reshape(105, 105, 1).astype(np.float64)


def get_metrics(y_true, y_pred, threshold_interval=0.001):
  labels = [0, 1]
  thresholds = np.arange(0, 1, threshold_interval)
  results = np.zeros((thresholds.size, 12))
  for i, threshold in enumerate(thresholds):
    preds = y_pred > threshold
    prc = precision(y_true, preds, labels)           # (positive predictive value)
    rec = recall(y_true, preds, labels)              # (true positive rate or sensitivity)
    spc = specificity(y_true, preds, labels)         # (true negative rate)
    f1 = f_score(y_true, preds, labels)              # p & r are of equal importance
    f2 = f_score(y_true, preds, labels, beta=2)      # recall is twice as important
    fh = f_score(y_true, preds, labels, beta=0.5)    # precision is twice as important
    acc = accuracy(y_true, preds, labels)

    results[i, 0] = threshold
    results[i, 1:5] = get_cm(y_true, preds, labels)
    for j, mt in enumerate([prc, rec, spc, f1, f2, fh, acc]):
      results[i, j+5] = mt
  return results


def print_res(thres, cm, metric, name):
  print(f"Highest {name}: {metric.max():.4f} at {thres[metric.argmax()]:.4f} threshold.")
  print("Confusion matrix:")
  print(cm[metric.argmax()].reshape((2, 2)))


def summary(metrics_data):
  thres = metrics_data[:, 0]
  cms = metrics_data[:, 1:5]

  for metric, metric_name in zip(metrics_data[:, 5:].T, METRICS):
    print_res(thres, cms, metric, metric_name)
    print("")


def load_images(pair_list_file, imgs_dir, preprocess=True, face_only=False):
  image_loader = ImageLoader(pair_list_file, imgs_dir, preprocess=preprocess, crop_face=face_only)
  pairs = []
  x_test = []
  y_true = []
  for pair, x, y in tqdm(image_loader, desc="Loading image pairs..."):
    pairs.append(pair)
    x_test.append(x)
    y_true.append(y)
  return pairs, [x_t for x_t in np.transpose(x_test, (1, 0, 2, 3, 4))], np.array(y_true)


def save_inference_data(pairs, y_true, y_pred, output_dir):
  with (output_dir / f"inference_data.csv").open("w") as f:
    f.write("img_pair,y_true,y_pred\n")
    for pair, y_t, y_p in zip(pairs, y_true, y_pred):
      f.write(f"{pair},{y_t},{y_p}\n")


def save_metrics_data(metrics_data, output_dir):
  with (output_dir / f"metrics_data.csv").open("w") as f:
    f.write("thres,tn,fp,fn,tp,prc,rec,spc,f1,f2,f.5,acc\n")
    for row in metrics_data:
      f.write(",".join(row.astype(str))+'\n')


def main_no_infer(args):
  output_dir = Path(args.output_dir)
  for out_dir in output_dir.iterdir():
    if out_dir.is_file():
      continue
    inference_data = out_dir / "inference_data.csv"
    if not inference_data.exists():
      print(f"No inference data in {str(out_dir)}")
      continue
    with inference_data.open("r") as f:
      lines = f.readlines()
    pairs = []
    y_true = []
    y_pred = []
    for line in lines[1:]:
      pair, y_t, y_p = line[:-1].split(",")
      pairs.append(pair)
      y_true.append(y_t)
      y_pred.append(y_p)
    metrics_data = get_metrics(np.array(y_true).astype(int), np.array(y_pred).astype(float), args.threshold_interval)

    print(f"\nSummary for weight {out_dir.name}")
    summary(metrics_data)

    if args.no_save:
      continue
    save_metrics_data(metrics_data, out_dir)
    for i, metric in enumerate(METRICS):
      plot_curve(metrics_data[:, 0], metrics_data[:, i+5], (out_dir/f"{metric}_curve.png"), ylabel=metric)


def main(args):
  data_dir = Path(args.dataset_dir)
  pair_list = Path(args.img_pair_list)
  weights_dir = Path(args.weights_dir)
  output_dir = Path(args.output_dir)

  net = Siamese()
  pairs, x_test, y_true = load_images(pair_list, data_dir, face_only=args.face_only)
  for weight in weights_dir.iterdir():
    net.set_weight(weight)
    y_pred = net.predict(x_test, batch_size=args.batch_size)
    metrics_data = get_metrics(y_true, y_pred, args.threshold_interval)

    print(f"\nSummary for weight {weight.name}")
    summary(metrics_data)

    if args.no_save:
      continue
    out_dir = output_dir / f"{weight.name}.bs_{args.batch_size}.faceonly_{int(args.face_only)}"
    out_dir.mkdir(exist_ok=True, parents=True)
    save_inference_data(pairs, y_true, y_pred, out_dir)
    save_metrics_data(metrics_data, out_dir)
    for i, metric in enumerate(METRICS):
      plot_curve(metrics_data[:, 0], metrics_data[:, i+5], (out_dir/f"{metric}_curve.png"), ylabel=metric)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-w", "--weights-dir", type=str, default="lfw2/weights", help="Saved weights directory")
  parser.add_argument("-d", "--dataset-dir", type=str, default="lfw2/lfw2", help="Dataset directory")
  parser.add_argument("-p", "--img-pair-list", type=str, default="lfw2/splits/test.txt", help="Text file containing the list of the image pairs")
  parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size for inference")
  parser.add_argument("-t", "--threshold-interval", type=float, default=0.001, help="Threshold interval")
  parser.add_argument("-o", "--output-dir", type=str, default="results", help="Directory to save the results")
  parser.add_argument("-f", "--face-only", action="store_true", help="Isolate the face before inference")
  parser.add_argument("--no-save", action="store_true", help="Don't save the results")
  parser.add_argument("--no-infer", action="store_true", help="Skip the inference process (the inference data must already in the result directory)")
  args = parser.parse_args()

  if args.no_infer:
    main_no_infer(args)
  else:
    main(args)
