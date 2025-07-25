import numpy as np
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
from collections import deque

class DynamicModelSelector:
    def __init__(self, model_dict, buffer_size=30, dtw_threshold=1.0, score_step=3):
        self.models = model_dict
        self.buffer = deque(maxlen=buffer_size)
        self.dtw_threshold = dtw_threshold
        self.score_step = score_step
        self.previous_best_model = None

    def _predict_all(self, sequence):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(sequence)
            results[name] = y_pred
        return results

    def run_all_model(self, true, preds):
        mse_scores = {name: mean_squared_error(true, pred) for name, pred in preds.items()}
        return min(mse_scores, key=mse_scores.get)

    def update_buffer(self, sequence, ground_truth, used_triplet):
        preds = self._predict_all(sequence)
        best_model = self.run_all_model(ground_truth, preds)
        self.previous_best_model = best_model

        if used_triplet and used_triplet['model'] == best_model:
            used_triplet['score'] += self.score_step
        elif used_triplet:
            used_triplet['score'] -= self.score_step

        self.buffer.append({
            'sequence': sequence,
            'model': best_model,
            'score': self.buffer.maxlen
        })

    def select_model(self, current_sequence):
        if not self.buffer:
            return list(self.models.keys())[0]  # fallback to first model

        # Find closest sequence in buffer
        distances = [(triplet, fastdtw(current_sequence, triplet['sequence'])[0])
                     for triplet in self.buffer]
        used_triplet, dist = min(distances, key=lambda x: x[1])

        # Decay scores
        for t in self.buffer:
            t['score'] -= 1

        # Select model
        if dist < self.dtw_threshold:
            return used_triplet['model'], used_triplet
        else:
            return self.previous_best_model, None  # fallback if no close match
