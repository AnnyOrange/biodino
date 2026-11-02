#!/usr/bin/env python3
"""Repackage existing HF parquet files for the unchanged FM data installer."""
import argparse,csv,hashlib,json,os,tempfile,zipfile
from pathlib import Path,PurePosixPath

def safe(root,name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe path: '+name)
    target=root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Escaping source: '+name)
    return target

def build(source,output,rows):
    if output.exists():raise FileExistsError(f'{output}: preserve existing file; verify it with the normal data acceptance flow')
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=output.name+'.',suffix='.partial',dir=output.parent);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            metadata=[]
            for row in rows:
                f=safe(source,row['remote_path']);safe(Path('/archive'),row['archive_member'])
                expected=int(row['bytes'])
                if f.stat().st_size!=expected:raise ValueError('Size mismatch: '+str(f))
                digest=hashlib.sha256();n=0
                with f.open('rb') as reader,z.open(row['archive_member'],'w',force_zip64=True) as writer:
                    for block in iter(lambda:reader.read(4*1024*1024),b''):
                        digest.update(block);n+=len(block);writer.write(block)
                if n!=expected or digest.hexdigest()!=row['sha256']:raise ValueError('SHA256 mismatch: '+str(f))
                metadata.append(dict(path=row['archive_member'],bytes=n,sha256=digest.hexdigest()))
            z.writestr('PACKAGE_FILES.json',json.dumps(metadata,indent=2))
        # Publish only a complete verified archive; do not overwrite any existing file.
        os.link(tmp,output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print('READY',output,len(rows),'verified members',flush=True)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source',type=Path,required=True,help='Local HF repo root containing dataset_transfer_formal_v3_remaining_20260920/')
    a.add_argument('--output',type=Path,required=True,help='Uploads directory used by install_data.py')
    a.add_argument('--mapping',type=Path,default=Path(__file__).parent/'existing_hf/PARQUET_REUSE_MEMBERS.csv')
    x=a.parse_args();groups={}
    with x.mapping.open() as f:
        for row in csv.DictReader(f):groups.setdefault(row['archive'],[]).append(row)
    for name,rows in groups.items():build(x.source,safe(x.output,name),rows)
if __name__=='__main__':main()
