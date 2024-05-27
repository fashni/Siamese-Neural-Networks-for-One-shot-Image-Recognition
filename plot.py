from itertools import cycle
import matplotlib.pyplot as plt

def plot_curve(px, py, save_dir="curve.png", xlabel="Threshold", ylabel="Metric"):
  fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

  markers = cycle("123x.os")
  markers_on = list(range(0, 1000, 20))
  marker_styles = dict(markevery=markers_on)

  bbox = dict(
    boxstyle="round",
    ec=(1., 0.5, 0.5),
    fc=(1., 0.8, 0.8),
    alpha=0.5,
  )

  x_best = px[py.argmax()]
  ax.plot(px, py, marker=next(markers), linewidth=1, color="blue", label=f'{ylabel}: {py.max():.4f} at {x_best:.4f}', **marker_styles)
  ax.axvline(x_best, color="red")
  ax.text(x_best+((2*(x_best<0.5)-1)*0.055), py.max(), f"x: {x_best:.4f}\ny: {py.max():.4f}", transform=ax.get_xaxis_transform(), color="red", bbox=bbox, ha="center", va="center")
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim(0, 1.1)
  ax.set_ylim(0, 1.1)
  ax.grid(True, "both", "both")
  ax.legend(shadow=True, fancybox=True, bbox_to_anchor=(0.5, -0.1), loc="upper center", ncols=2)
  fig.savefig(save_dir, dpi=250)
  plt.close()

def plot_roc(fpr, tpr, auc, save_dir="ROC_Curve.png"):
  fig, ax = plt.subplots(1, 1, figsize=(7, 6), tight_layout=True)
  ax.set_title(f"ROC Curve, AUC = {auc:.4f}")
  ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
  ax.plot([0, 1], [0, 1], 'k--', lw=1)
  ax.set_ylabel('Recall')
  ax.set_xlabel('1 - Specificity')
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1.05])
  ax.tick_params()
  ax.grid(True)
  ax.legend(shadow=True, fancybox=True, loc=4)
  fig.savefig(save_dir, dpi=250)
  plt.close()

