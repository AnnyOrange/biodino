import unittest
from selected29_metrics_20261019 import COUNTS, PANNUKE, dense_mean, family_means, metric


class Selected29Tests(unittest.TestCase):
    def test_nested_retrieval_uses_global_not_cell_type(self):
        result = {'rows': [dict(task='retrieval', aggregation='HUVEC', map_at_5=.9),
                           dict(task='retrieval', aggregation='global', map_at_5=.4),
                           dict(task='clustering', nmi=.7)]}
        self.assertEqual(metric('retrieval', result), .4)
        result['rows'].pop(1)
        with self.assertRaises(ValueError):
            metric('retrieval', result)

    def test_pannuke_needs_every_rotation_and_seed(self):
        folds = sorted(PANNUKE)
        results = {fold: {0: value, 1: value, 2: value} for fold, value in zip(folds, (.3, .6, .9))}
        self.assertAlmostEqual(dense_mean('pannuke', results), .6)
        del results[folds[0]][2]
        self.assertIsNone(dense_mean('pannuke', results))
        results[folds[0]][2] = .3
        del results[folds[1]]
        self.assertIsNone(dense_mean('pannuke', results))

    def test_equal_family_weight_and_missing_dataset(self):
        selected = {(f, str(i)) for f, n in COUNTS.items() for i in range(n)}
        cells = {f: {str(i): float(f == 'classification') for i in range(n)} for f, n in COUNTS.items()}
        self.assertAlmostEqual(family_means(cells, selected)['overall'], .2)
        del cells['segmentation']['0']
        self.assertIsNone(family_means(cells, selected)['overall'])


if __name__ == '__main__':
    unittest.main()
