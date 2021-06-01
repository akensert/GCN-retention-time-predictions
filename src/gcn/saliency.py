import numpy as np
import tensorflow as tf
import os
import glob
from rdkit.Chem import Draw


class Saliency():

    def __init__(self, import_dir, signature='serving_default'):
        self._loaded = tf.saved_model.load(import_dir)
        self._predict = self._loaded.signatures[signature]
        self._loss = tf.compat.v1.losses.huber_loss

    def atom_importance(self, A, H, y, reduce=True):

        # remove potential padding
        keep_idx = np.where(A.sum(axis=1) != 0)[0]
        H = tf.convert_to_tensor(H[keep_idx])
        A = tf.convert_to_tensor(A[keep_idx][:, keep_idx])

        y = tf.convert_to_tensor(y)[tf.newaxis]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(H)
            y_pred = self._predict(**{'A': A, 'H': H})['prediction']
            loss = self._loss(y, y_pred)

        gradients = tape.gradient(loss, H)
        gradients = tf.abs(gradients)

        if reduce:
            gradients = tf.reduce_sum(gradients, axis=1)

        return gradients.numpy()

    def bond_importance(self, A, H, y):

        y = tf.convert_to_tensor(y)[tf.newaxis]

        idx_nonzero = tf.where(A > 0)
        A_nonzero = tf.gather_nd(A, idx_nonzero)
        with tf.GradientTape() as tape:
            tape.watch(A_nonzero)
            A = tf.scatter_nd(idx_nonzero, A_nonzero, shape=A.shape)
            y_pred = self._predict(**{'A': A, 'H': H})['prediction']
            loss = self._loss(y, y_pred)

        gradients = tape.gradient(loss, A_nonzero)
        gradients = tf.abs(gradients)
        return gradients#tf.math.segment_sum(gradients, idx_nonzero[:, 0])

    @staticmethod
    def draw_atom_saliency_on_mol(mol, saliency, path, size=(1000, 1000)):

        if not os.path.isdir('/'.join(path.split('/')[:-1])):
            os.makedirs('/'.join(path.split('/')[:-1]))

        drawer = Draw.MolDraw2DCairo(*size)
       # drawer.drawOptions().scaleBondWidth = True
        drawer.drawOptions().bondLineWidth = 3

        saliency = saliency / saliency.max()

        Draw.SimilarityMaps.GetSimilarityMapFromWeights(
            mol=mol,
            weights=[float(s) for s in saliency],
            size=size,
            coordScale=1.0,
            colors='g',
            alpha=0.4,
            contourLines=10,
            draw2d=drawer);

        drawer.FinishDrawing()
        drawer.WriteDrawingText(path)
