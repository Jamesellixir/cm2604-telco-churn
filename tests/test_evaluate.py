import unittest
import numpy as np
from src.evaluate import metrics, group_errors

class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_counts_and_metrics(self):
        y=np.array([0,0,1,1]); p=np.array([0,1,0,1]); probabilities=np.array([.1,.8,.3,.9])
        result=metrics(y,p,probabilities)
        for key in ["accuracy","balanced_accuracy","precision","recall","f1"]:
            self.assertAlmostEqual(result[key],.5)
        self.assertAlmostEqual(result["roc_auc"],.75)
        errors=group_errors(y,p)
        self.assertEqual([errors[k] for k in ["tn","fp","fn","tp"]],[1,1,1,1])
        self.assertEqual(errors["false_negative_rate"],.5)
    def test_absent_class_rates_are_undefined(self):
        errors=group_errors(np.array([0,0]),np.array([0,1]))
        self.assertIsNone(errors["false_negative_rate"])
        self.assertEqual(errors["false_positive_rate"],.5)
