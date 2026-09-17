import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_hest_patient_groups import audit_patient_groups, verify_metadata, write_checked_preflight


class HESTPatientGroupsTest(unittest.TestCase):
    def setUp(self):
        self.payload = b'id,patient,dataset_title\nA,Patient 1,Cohort\nB,Patient 2,Cohort\nTENX111,,Colon\n'
        blob = hashlib.sha1(b'blob ' + str(len(self.payload)).encode() + b'\0' + self.payload).hexdigest()
        self.patch = patch('scripts.audit_hest_patient_groups.OFFICIAL_GIT_BLOB', blob)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.preflight = {
            'status': 'PASS', 'payload_verified': True, 'release_status': 'PASS', 'release_files': 212,
            'tasks': {'COAD': {'folds': {'0': {'train': {'sample_ids': ['A']},
                                            'test': {'sample_ids': ['B', 'TENX111']}}}}},
        }

    def test_one_released_unknown_is_explicit_not_fabricated(self):
        result = audit_patient_groups(self.preflight, self.payload)
        self.assertEqual(result['status'], 'PASS_KNOWN_PATIENTS_ONE_UNRESOLVED')
        self.assertEqual(result['known_slides'], 2)
        self.assertEqual(result['unknown_ids'], ['TENX111'])
        self.assertFalse(result['patient_disjoint_fully_verified'])

    def test_changed_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, 'official Git blob'):
            verify_metadata(self.payload.replace(b'Patient 2', b'Patient 1'))

    def test_known_patient_overlap_fails(self):
        payload = self.payload.replace(b'Patient 2', b' patient1')
        blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        with patch('scripts.audit_hest_patient_groups.OFFICIAL_GIT_BLOB', blob):
            result = audit_patient_groups(self.preflight, payload)
        self.assertEqual(result['status'], 'FAIL_KNOWN_PATIENT_OVERLAP')
        self.assertEqual(result['failures'][0]['patient_overlap'], ['COAD:patient1'])

    def test_generic_labels_scoped_by_tissue(self):
        self.preflight['tasks']['PRAD'] = self.preflight['tasks']['COAD']
        result = audit_patient_groups(self.preflight, self.payload)
        self.assertEqual(result['known_slides'], 4)
        self.assertEqual({row['patient_key'] for row in result['sample_patient_mapping'] if row['patient_key']},
                         {'COAD:patient1', 'COAD:patient2', 'PRAD:patient1', 'PRAD:patient2'})

    def test_preserves_full_payload_preflight_and_companion_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / 'preflight.json'
            metadata = root / 'source.csv'
            preflight.write_text(json.dumps(self.preflight))
            metadata.write_bytes(self.payload)
            write_checked_preflight(preflight, metadata)
            checked = json.loads((root / 'preflight_patient_checked.json').read_text())
            self.assertEqual({key: value for key, value in checked.items() if key != 'patient_grouping'}, self.preflight)
            self.assertTrue(checked['payload_verified'])
            self.assertEqual(checked['release_files'], 212)
            self.assertEqual((root / 'patient_metadata.csv').read_bytes(), self.payload)
            self.assertEqual(json.loads((root / 'patient_grouping.json').read_text()), checked['patient_grouping'])

    def test_unreleased_missing_patient_does_not_pass(self):
        payload = self.payload.replace(b'B,Patient 2', b'B,')
        blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        with patch('scripts.audit_hest_patient_groups.OFFICIAL_GIT_BLOB', blob):
            result = audit_patient_groups(self.preflight, payload)
        self.assertEqual(result['status'], 'FAIL_UNRESOLVED_PATIENT_IDENTITIES')


if __name__ == '__main__':
    unittest.main()
