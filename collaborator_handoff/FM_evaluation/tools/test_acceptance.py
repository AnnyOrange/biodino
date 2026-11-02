import importlib.util,json,tempfile,unittest,hashlib
from pathlib import Path
s=importlib.util.spec_from_file_location('accept',Path(__file__).with_name('accept.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
s=importlib.util.spec_from_file_location('install',Path(__file__).with_name('install_data.py'));i=importlib.util.module_from_spec(s);s.loader.exec_module(i)
class AcceptanceTests(unittest.TestCase):
 def test_corrupt_source_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'image';p.write_bytes(b'original');h=a.sha(p);a.verify_file(p,h);p.write_bytes(b'corrupt')
   with self.assertRaises(ValueError):a.verify_file(p,h)
 def test_missing_prediction_fails(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);p=r/'result.json';p.write_text(json.dumps(dict(model='biodino_7b',seed=2,prediction_path=str(r/'missing.npz'))))
   with self.assertRaises(ValueError):a.validate_result(r,dict(result=p.name,task='segmentation',seed=2),'/weights/test')
 def test_wrong_seed_fails(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'r.json').write_text(json.dumps(dict(model='biodino_7b',seed=0)))
   with self.assertRaises(ValueError):a.validate_result(r,dict(result='r.json',task='segmentation',seed=1),'/weights/test')
 def test_valid_classification_then_missing_prediction(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);pred=r/'pred.csv';pred.write_text('id,score\n1,0.5\n')
   p=r/'r.json';p.write_text(json.dumps(dict(model='biodino_7b',balanced_accuracy=0.5,prediction_path=str(pred))))
   job=dict(result='r.json',task='classification',dataset='bloodmnist')
   self.assertTrue(a.validate_result(r,job,'/weights/test')['artifacts'])
   pred.unlink()
   with self.assertRaises(ValueError):a.validate_result(r,job,'/weights/test')
 def test_nan_fails(self):
  with self.assertRaises(ValueError):a.finite_tree({'metrics':{'score':float('nan')}})
 def test_traversal_fails(self):
  for name in ['../escape','/absolute','a/../../escape']:
   with self.assertRaises(ValueError):i.safe_target(Path('/tmp/allowed'),name)
 def test_empty_result_fails(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'r.json').write_text('{}')
   with self.assertRaises(ValueError):a.validate_result(r,dict(result='r.json',task='classification'),'/weights/test')
if __name__=='__main__':unittest.main()
