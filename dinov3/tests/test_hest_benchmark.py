from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from dinov3.eval.bio_frozen_eval.hest import (
    HESTPatchDataset, inspect_hest, load_hest_targets, run_hest_probe_split,
)


def write_expression(path, encoding="dense"):
    from scipy import sparse

    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    with h5py.File(path, "w") as handle:
        for key, names in (("obs", ["c", "a", "b"]), ("var", ["g2", "g3", "g1"])):
            group = handle.create_group(key)
            group.attrs["_index"] = "identity"
            group.create_dataset("identity", data=np.array(names, dtype="S"))
        if encoding == "dense":
            handle.create_dataset("X", data=matrix)
        else:
            value = sparse.csr_matrix(matrix) if encoding == "csr_matrix" else sparse.csc_matrix(matrix)
            group = handle.create_group("X")
            group.attrs["encoding-type"] = encoding
            for key in ("data", "indices", "indptr"):
                group.create_dataset(key, data=getattr(value, key))


class HESTBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / "IDC"
        for name in ("splits", "patches", "adata"):
            (self.task / name).mkdir(parents=True)
        (self.task / "var_50genes.json").write_text(json.dumps({"genes": ["g1", "g2"]}))
        for split in ("train", "test"):
            with h5py.File(self.task / "patches" / f"{split}.h5", "w") as handle:
                handle.create_dataset("barcode", data=np.array([[b"b"], [b"a"]]))
                handle.create_dataset("img", data=np.full((2, 8, 8, 3), 123, dtype=np.uint8))
            write_expression(self.task / "adata" / f"{split}.h5ad")
            with (self.task / "splits" / f"{split}_0.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "patches_path", "expr_path"])
                writer.writeheader()
                writer.writerow({"sample_id": split, "patches_path": f"patches/{split}.h5", "expr_path": f"adata/{split}.h5ad"})

    def test_targets_align_barcode_and_gene_order_for_all_storage_types(self):
        for encoding in ("dense", "csr_matrix", "csc_matrix"):
            path = self.root / f"{encoding}.h5ad"
            write_expression(path, encoding)
            target = load_hest_targets(path, ["g1", "g2"], ["b", "a"])
            np.testing.assert_allclose(target, np.log1p([[9, 7], [6, 4]]))

    def test_missing_and_duplicate_identities_fail_closed(self):
        path = self.task / "adata/train.h5ad"
        with self.assertRaisesRegex(ValueError, "Missing"):
            load_hest_targets(path, ["g1"], ["missing"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_hest_targets(path, ["g1"], ["a", "a"])

    def test_patch_dataset_has_original_pixels_and_aligned_targets(self):
        dataset = HESTPatchDataset(self.task)
        image, target, path = dataset[0]
        self.assertEqual(len(dataset), 2)
        self.assertEqual(image.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(image), 123)
        np.testing.assert_allclose(target, np.log1p([9, 7]))
        self.assertIn("::train::b", path)
        self.assertTrue(dataset.protocol_complete)
        self.assertFalse(HESTPatchDataset(self.task, max_samples=1).protocol_complete)

    def test_preflight_records_hashes_without_patient_claim(self):
        report = inspect_hest(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["patient_disjoint_verified"])
        self.assertEqual(len(report["tasks"]["IDC"]["gene_sha256"]), 64)
        self.assertEqual(report["tasks"]["IDC"]["folds"]["0"]["train"]["spots"], 2)

    def test_preflight_rejects_shared_sample_or_asset(self):
        path = self.task / "splits/test_0.csv"
        path.write_text("sample_id,patches_path,expr_path\ntrain,patches/train.h5,adata/train.h5ad\n")
        report = inspect_hest(self.root)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("leakage", report["tasks"]["IDC"]["errors"][0])

    def test_preflight_rejects_unpaired_folds_and_missing_variable_genes(self):
        (self.task / "splits/train_1.csv").write_text((self.task / "splits/train_0.csv").read_text())
        self.assertEqual(inspect_hest(self.root)["status"], "FAIL")
        (self.task / "var_50genes.json").unlink()
        self.assertIn("variable-gene", inspect_hest(self.root)["tasks"]["IDC"]["errors"][0])

    def test_ridge_matches_official_train_only_reference(self):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(4)
        train = rng.normal(size=(32, 6))
        test = rng.normal(size=(8, 6)) + 10
        y_train, y_test = train[:, :2] + 5, test[:, :2] + 5
        result = run_hest_probe_split(train, y_train, test, y_test, latent_dim=4, seed=1)
        transform = make_pipeline(StandardScaler(), PCA(n_components=4, random_state=1))
        x = transform.fit_transform(train)
        reference = Ridge(alpha=100 / (4 * 2), solver="lsqr", fit_intercept=False,
                          max_iter=1000, random_state=1).fit(x, y_train).predict(transform.transform(test))
        np.testing.assert_allclose(result["predictions"], reference)
        self.assertEqual(result["hyperparameters"]["alpha"], 12.5)
        self.assertEqual(result["protocol_decision"], "PROPOSED_BY_US")

    def test_no_silent_pca_cap_or_undefined_gene_omission(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(12, 3))
        with self.assertRaisesRegex(ValueError, "no silent reduction"):
            run_hest_probe_split(x, x[:, :1], x[:4], x[:4, :1])
        result = run_hest_probe_split(x, np.ones((12, 1)), x[:4], np.ones((4, 1)), latent_dim=2)
        self.assertFalse(result["metric_valid"])
        self.assertTrue(np.isnan(result["metrics"]["gene_wise_pearson"]))


if __name__ == "__main__":
    unittest.main()
