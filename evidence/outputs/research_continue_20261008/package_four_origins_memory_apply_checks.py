"""Memory-filesystem-only packager mutation tests; never touch real delivery."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
SOURCE='src/package_research_four_origins_20261008.py'
spec=importlib.util.spec_from_file_location('four_memory_packager',c.safe(SOURCE))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
c.need(c.sha(SOURCE)=='a7edf6d9e9d347b7afc3fbb8c59d9be7db24415dae32a7679ce5bec212e5f16a','Fixed packager')


class MemoryFS:
    def __init__(self):self.files={};self.writes=0;self.fail_at=None
    def path(self,name):return MemoryPath(self,str(PurePosixPath(name)))


class MemoryPath:
    def __init__(self,fs,name):self.fs,self.key=fs,name
    def __truediv__(self,value):return self.fs.path(str(PurePosixPath(self.key)/str(value)))
    def __str__(self):return self.key
    def __eq__(self,other):return isinstance(other,MemoryPath) and other.fs is self.fs and other.key==self.key
    def __hash__(self):return hash(self.key)
    @property
    def parent(self):return self.fs.path(str(PurePosixPath(self.key).parent))
    @property
    def parents(self):return tuple(self.fs.path(str(x)) for x in PurePosixPath(self.key).parents)
    @property
    def name(self):return PurePosixPath(self.key).name
    def with_name(self,name):return self.parent/name
    def relative_to(self,root):return PurePosixPath(self.key).relative_to(PurePosixPath(str(root)))
    def is_file(self):return self.key in self.fs.files
    def exists(self):return self.is_file() or any(x.startswith(self.key+'/') for x in self.fs.files)
    def is_symlink(self):return False
    def is_junction(self):return False
    def read_bytes(self):return self.fs.files[self.key]
    def stat(self):return SimpleNamespace(st_size=len(self.read_bytes()))
    def mkdir(self,**kwargs):pass
    def open(self,mode):
        if mode=='rb':return io.BytesIO(self.read_bytes())
        if mode!='xb':raise AssertionError('Unexpected memory mode '+mode)
        if self.key in self.fs.files:raise FileExistsError(self.key)
        self.fs.files[self.key]=b'';self.fs.writes+=1;path=self
        class Writer(io.BytesIO):
            def write(self,value):
                if path.fs.fail_at==path.fs.writes:raise RuntimeError('injected memory partial write')
                return super().write(value)
            def close(self):
                if not self.closed:path.fs.files[path.key]=self.getvalue()
                super().close()
        return Writer()


def fixture():
    fs=MemoryFS();root=fs.path('/workspace');repo=root/'deliverables/v22_github'
    names=set(p.NAVIGATION)|set(p.PRODUCTION)|{'research/source_manifest.json'}
    history=[{'copy':f'research/history_{i:03}.py','sha256':p.sha_bytes(b'pass\n')} for i in range(379)]
    names.update(x['copy'] for x in history);i=0
    while len(names)<3229:names.add(f'evidence/old_{i:04}.json');i+=1
    for name in names:fs.files[str(repo/name)]=b'pass\n'
    for name in p.NAVIGATION:fs.files[str(repo/name)]=('original '+name+' body\n').encode()
    fs.files[str(repo/'research/source_manifest.json')]=p.encoded(history)
    old=[{'path':name,'bytes':len(fs.files[str(repo/name)]),'sha256':p.sha_bytes(fs.files[str(repo/name)])} for name in sorted(names)]
    manifest=p.encoded({'files':old});fs.files[str(repo/'SOURCE_MANIFEST.json')]=manifest
    entries=[]
    for name,value in [('src/round114_research.py',b'pass # new source\n'),
        ('outputs/research_round117_v2_20261008/postfit_review.json',b'{"synthetic":true}\n')]:
        fs.files[str(root/name)]=value
        entries.append({'source':name,'destination':p.destination(name),'bytes':len(value),'sha256':p.sha_bytes(value)})
    return fs,root,repo,manifest,entries


def contexts(fs,root,repo,manifest,entries):
    def safe(base,name):p.relative(name);return base/name
    def replace(source,target):fs.files[str(target)]=fs.files.pop(str(source))
    return [patch.object(p,'ROOT',root),patch.object(p,'REPO',repo),patch.object(p,'safe',side_effect=safe),
        patch.object(p,'BASE_MANIFEST_SHA256',p.sha_bytes(manifest)),
        patch.object(p,'check_git_base',return_value={'HEAD':p.BASE_HEAD}),
        patch.object(p,'inventory',return_value=(entries,{n:'combined_gate_failed_no_release' for n in p.ROUNDS},{'synthetic':True})),
        patch.object(p,'summary',return_value=b'# 2026-10-08 synthetic summary\n'),patch.object(p.os,'replace',side_effect=replace)]


def run():
    from contextlib import ExitStack
    checks=[]
    fs,root,repo,manifest,entries=fixture()
    with ExitStack() as stack:
        for context in contexts(fs,root,repo,manifest,entries):stack.enter_context(context)
        result=p.run({n:{'no_release_reason':'synthetic fixed gate failed'} for n in p.ROUNDS},[],True,{},False)
        c.need(result['status']=='copied_exact_bytes_local_only','Memory apply success')
        before=p.json_value((repo/p.COPY_MANIFEST).read_bytes())
        c.need(len(before['protected_before'])==3226,'All old memory records protected')
        for name in p.NAVIGATION:fs.files[str(repo/name)]=(repo/name).read_bytes()+b'2026-10-08 dated addition\n'
        completed=p.finish_manifest()
        c.need(completed['protected_files_unchanged'] and len(p.json_value((repo/'SOURCE_MANIFEST.json').read_bytes())['files'])==3238,
               'Memory exact manifest membership after finish')
        checks.append('complete_memory_apply_refresh_dated_navigation_finish_exact3238_members')
        try:p.finish_manifest()
        except ValueError:checks.append('second_memory_finish_refuses_overwrite')
        else:raise AssertionError('Second finish overwrote receipt')
    fs,root,repo,manifest,entries=fixture();fs.fail_at=2
    with ExitStack() as stack:
        for context in contexts(fs,root,repo,manifest,entries):stack.enter_context(context)
        try:p.run({},[],True,{},False)
        except RuntimeError:c.need(fs.writes==2,'Exact injected failure position')
        else:raise AssertionError('No partial failure')
        c.need((repo/entries[0]['destination']).read_bytes()==(root/entries[0]['source']).read_bytes()
               and (repo/entries[1]['destination']).exists(),'Partial evidence preserved')
        checks.append('injected_memory_copy_failure_preserves_complete_and_partial_outputs')
        before=dict(fs.files);fs.fail_at=None
        try:p.run({},[],True,{},False)
        except ValueError:c.need(fs.files==before,'Rerun must not alter partial memory outputs');checks.append('partial_memory_rerun_refuses_unlisted_destinations_without_writes')
        else:raise AssertionError('Partial rerun accepted')
    c.need(c.sha(SOURCE)=='a7edf6d9e9d347b7afc3fbb8c59d9be7db24415dae32a7679ce5bec212e5f16a','Final source unchanged')
    result={'status':'passed_memory_filesystem_only','checks':checks,'source':c.record(SOURCE),'review_tool':c.record(__file__),
        'actual_deliverables_FS_reads_or_writes':False,'actual_copy_or_formal_package_check':False,'GPU_work':False,'new_model_fits':0}
    c.write('outputs/research_continue_20261008/package_four_origins_memory_review.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':run()
