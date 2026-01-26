from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EvaluationMetrics:
    """
    Data Transfer Object for model evaluation results.
    """

    accuracy: float
    loss: float
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def confusion_matrix(self) -> NDArray[np.int_]:
        """
        Returns the confusion matrix as a 2x2 NumPy array:
        [[TN, FP],
         [FN, TP]]
        """
        return np.array(
            [
                [self.tn, self.fp],
                [self.fn, self.tp],
            ]
        )
