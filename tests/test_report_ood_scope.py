import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'report_ood_scope', Path(__file__).resolve().parents[1] / 'scripts/report_ood_scope.py')
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


class ScopeReportTest(unittest.TestCase):
    def test_scope_preserves_missing_and_partial_evidence_and_unknown_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            rows = [
                dict(model='bio', domain='OOD', dataset='xray', value='0.98765432109876543', metric_status='complete'),
                dict(model='bio', domain='OOD', dataset='cryo', value='0.95', metric_status='complete'),
                dict(model='fm', domain='OOD', dataset='film_age', value='0.6', metric_status='partial'),
                dict(model='bio', domain='OOD', dataset='film_age', value='', metric_status='missing'),
                dict(model='bio', domain='OOD', dataset='new_source', value='0.8', metric_status='complete'),
                dict(model='bio', domain='ID', dataset='id_dataset', value='0.9', metric_status='complete'),
            ]
            source = report / 'paper_results.csv'
            source.write_text(scope.csv_text(rows, list(rows[0])))
            original = source.read_bytes()
            summary = scope.refresh(report)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(summary['indexed_dataset_families'], 4)
            self.assertEqual(summary['ood_detection_dataset_families'], 2)
            self.assertEqual(summary['native_downstream_dataset_families'], 1)
            self.assertEqual(summary['unclassified_dataset_families'], ['new_source'])
            self.assertFalse(summary['overall_ood_generalization_claim_supported'])
            film = next(d for d in summary['datasets'] if d['dataset'] == 'film_age')
            self.assertEqual((film['numeric_metric_rows'], film['complete_metric_rows'],
                              film['partial_metric_rows'], film['missing_metric_rows']), (1, 0, 1, 1))
            exported = list(csv.DictReader((report / 'ood_results_scoped.csv').read_text().splitlines()))
            self.assertEqual([{k: r[k] for k in rows[0]} for r in exported], rows[:-1])
            self.assertEqual(exported[-1]['evaluation_role'], 'unclassified')
            self.assertIn('no_generalization_claim', exported[-1]['inference_scope'])
            self.assertEqual(json.loads((report / 'ood_scope_status.json').read_text()), summary)


if __name__ == '__main__':
    unittest.main()
