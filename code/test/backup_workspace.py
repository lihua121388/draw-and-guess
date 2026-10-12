"""Complete project TAR snapshot with content hashes, hardlinks and read-back verification."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import time


class HashReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256()
    def read(self,n=-1):
        b=self.stream.read(n);self.digest.update(b);return b


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
    return digest.hexdigest()


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--name',default='full_project_20261013');args=p.parse_args();root=args.root.resolve();dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
    final=dest/(args.name+'.tar');partial=dest/(args.name+'.tar.partial');manifest_path=dest/(args.name+'.manifest.json')
    if final.exists() or partial.exists() or manifest_path.exists():raise FileExistsError('Backup names exist; never overwrite backups')
    # Only backup-generated record directory and new Git metadata are omitted.
    excluded={'.git','backup_records'}
    paths=sorted((x for x in root.rglob('*') if x.relative_to(root).parts[0] not in excluded),key=lambda x:x.relative_to(root).as_posix())
    physical=set();bytes_unique=0;logical=0
    for path in paths:
        if path.is_file() and not path.is_symlink():
            st=path.stat();logical+=st.st_size;key=(st.st_dev,st.st_ino)
            if key not in physical:bytes_unique+=st.st_size;physical.add(key)
    if shutil.disk_usage(dest).free<bytes_unique+2*1024**3:raise RuntimeError('Destination needs unique content size plus2GiB free')
    rows=[];digests={};start=time.perf_counter();prefix=root.name
    print(f'Snapshot {len(paths)} entries, unique={bytes_unique/1024**3:.2f}GiB, logical={logical/1024**3:.2f}GiB',flush=True)
    with tarfile.open(partial,'w',format=tarfile.PAX_FORMAT,dereference=False) as tar:
        tar.addfile(tar.gettarinfo(str(root),arcname=prefix))
        for index,path in enumerate(paths):
            rel=path.relative_to(root).as_posix();arc=prefix+'/'+rel;info=tar.gettarinfo(str(path),arcname=arc)
            if info is None:raise RuntimeError('Unsupported file type '+rel)
            row=dict(path=rel,archive_name=arc,type='directory',size=0)
            if info.isfile():
                before=path.stat()
                with path.open('rb') as stream:
                    reader=HashReader(stream);tar.addfile(info,reader);digest=reader.digest.hexdigest()
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during backup: '+rel)
                digests[(before.st_dev,before.st_ino)]=digest;row.update(type='file',size=before.st_size,sha256=digest,mtime_ns=before.st_mtime_ns)
            elif info.islnk():
                st=path.stat();digest=digests[(st.st_dev,st.st_ino)];tar.addfile(info);row.update(type='hardlink',size=st.st_size,sha256=digest,target=info.linkname,mtime_ns=st.st_mtime_ns)
            elif info.issym():tar.addfile(info);row.update(type='symlink',target=info.linkname)
            elif info.isdir():tar.addfile(info)
            else:raise RuntimeError('Unsupported special file '+rel)
            rows.append(row)
            if (index+1)%2000==0:print(f'Archive {index+1}/{len(paths)} elapsed={time.perf_counter()-start:.0f}s',flush=True)
        manifest=dict(root=str(root),prefix=prefix,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            logical_bytes=logical,unique_bytes=bytes_unique,exclusions=sorted(excluded),
            exclusion_reason='Git initialized AFTER snapshot; backup_records is generated backup bookkeeping only.',entries=rows)
        payload=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).encode('utf-8');header=tarfile.TarInfo(prefix+'/BACKUP_CONTENT_MANIFEST.json');header.size=len(payload)
        tar.addfile(header,io.BytesIO(payload))
    expected={r['archive_name']:r for r in rows};checked={};seen=set();files_checked=0
    with tarfile.open(partial,'r:') as tar:
        for index,member in enumerate(tar):
            if member.name==prefix or member.name==prefix+'/BACKUP_CONTENT_MANIFEST.json':continue
            row=expected.get(member.name)
            if row is None or member.name in seen:raise RuntimeError('Unexpected/duplicate archive member')
            seen.add(member.name)
            if member.isfile():
                digest=hashlib.sha256();stream=tar.extractfile(member)
                for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
                if digest.hexdigest()!=row['sha256']:raise RuntimeError('Read-back checksum mismatch '+member.name)
                checked[member.name]=digest.hexdigest();files_checked+=1
            elif member.islnk():
                if member.linkname!=row['target'] or checked.get(member.linkname)!=row['sha256']:raise RuntimeError('Hardlink verification mismatch')
                checked[member.name]=row['sha256'];files_checked+=1
            elif member.issym():
                if member.linkname!=row['target']:raise RuntimeError('Symlink mismatch')
            elif not member.isdir():raise RuntimeError('Unexpected special member')
            if index%2000==0:print(f'Verify {index}/{len(paths)}',flush=True)
    if seen!=set(expected):raise RuntimeError('Archive entries incomplete')
    # Check source membership and metadata did not change while snapshot was taken.
    current={x.relative_to(root).as_posix() for x in root.rglob('*') if x.relative_to(root).parts[0] not in excluded}
    if current!={r['path'] for r in rows}:raise RuntimeError('Workspace membership changed during snapshot')
    for row in rows:
        if row['type'] in ('file','hardlink'):
            st=(root/row['path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(row['size'],row['mtime_ns']):raise RuntimeError('Source changed during snapshot')
    archive_sha=file_sha(partial);os.replace(partial,final)
    save(manifest_path,manifest)
    result=dict(status='verified',archive=str(final),archive_bytes=final.stat().st_size,archive_sha256=archive_sha,
        manifest=str(manifest_path),manifest_sha256=file_sha(manifest_path),file_entries_verified=files_checked,
        directory_entries=sum(r['type']=='directory' for r in rows),source_membership_unchanged=True,
        seconds=time.perf_counter()-start,environment_note='Project files only; external Anaconda installation is not copied.')
    save(dest/(args.name+'.verification.json'),result)
    records=root/'backup_records';records.mkdir(exist_ok=True);save(records/(args.name+'.verification.json'),result)
    (dest/(args.name+'.sha256')).write_text(archive_sha+'  '+final.name+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
