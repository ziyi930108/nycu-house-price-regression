"""Synthetic-only packager audit: no real check/apply/copy/Git or ML data."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
SOURCE='src/package_research_four_origins_20261008.py'
SOURCE_SHA='a7edf6d9e9d347b7afc3fbb8c59d9be7db24415dae32a7679ce5bec212e5f16a'


def run():
    tick=time.monotonic();c.need(c.sha(SOURCE)==SOURCE_SHA,'Final packager source changed')
    spec=importlib.util.spec_from_file_location('independent_four_packager',c.safe(SOURCE))
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    checks=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):checks.append(name);return
        raise AssertionError('Accepted invalid '+name)
    def mark(name,ok):c.need(bool(ok),name);checks.append(name)
    def forbid(*args,**kwargs):raise AssertionError('No filesystem evidence or mutation allowed in pure author suite')
    with patch.object(Path,'read_bytes',side_effect=forbid),patch.object(Path,'read_text',side_effect=forbid),\
         patch.object(Path,'open',side_effect=forbid),patch.object(Path,'mkdir',side_effect=forbid),\
         patch.object(Path,'glob',side_effect=forbid),patch.object(Path,'rglob',side_effect=forbid),\
         patch.object(p.os,'replace',side_effect=forbid):
        author=p.self_check()
    mark('135_author_pure_memory_cases_under_reader_writer_traps',author['status']=='passed' and author['check_count']==135)
    # Full old-manifest comparison logic against a complete memory-only fake.
    names=set(p.NAVIGATION)|set(p.PRODUCTION)|{'research/source_manifest.json'}
    history=[{'copy':f'research/history_{i:03}.py','sha256':'b'*64} for i in range(379)]
    names.update(x['copy'] for x in history)
    i=0
    while len(names)<3229:names.add(f'evidence/old_{i:04}.json');i+=1
    records={name:{'path':name,'bytes':7,'sha256':'b'*64} for name in names}
    template={'files':list(records.values())};old_bytes=p.encoded(template)
    data={'SOURCE_MANIFEST.json':old_bytes,'research/source_manifest.json':p.encoded(history)}
    class MemoryPath:
        def __init__(self,name=''):self.name=name
        def __truediv__(self,part):return MemoryPath((self.name+'/' if self.name else '')+str(part))
        def read_bytes(self):return data[self.name]
    with patch.object(p,'REPO',MemoryPath()),patch.object(p,'BASE_MANIFEST_SHA256',p.sha_bytes(old_bytes)),\
         patch.object(p,'check_git_base',return_value={'HEAD':p.BASE_HEAD}):
        old,protected=p.protect_existing(records)
        mark('all3229_manifest_records_and3226_nonnav_members_memory_checked',len(old['files'])==3229 and len(protected)==3226)
        for kind in ('bytes','sha256','missing','history'):
            damaged=copy.deepcopy(records);name='evidence/old_0000.json'
            if kind=='bytes':damaged[name]['bytes']=8
            elif kind=='sha256':damaged[name]['sha256']='c'*64
            elif kind=='missing':damaged.pop(name)
            else:damaged[history[0]['copy']]['sha256']='c'*64
            reject('old_member_'+kind,lambda damaged=damaged:p.protect_existing(damaged))
    # Native Git-index binary parser, using only in-memory bytes.
    def index_payload(name,flags=0):
        encoded=name.encode();entry=b'\0'*60+struct.pack('!H',len(encoded)|flags)+encoded+b'\0'
        entry+=b'\0'*((-len(entry))%8)
        head=b'DIRC'+struct.pack('!II',2,1)+entry
        return head+p.hashlib.sha1(head).digest()
    mark('synthetic_index_checksum_membership_parser',p.git_index_paths(index_payload('README.md'))==['README.md'])
    reject('synthetic_index_checksum_corruption',lambda:p.git_index_paths(index_payload('README.md')[:-1]+b'x'))
    reject('synthetic_index_unmerged_stage',lambda:p.git_index_paths(index_payload('README.md',0x1000)))
    reject('synthetic_index_traversal_name',lambda:p.git_index_paths(index_payload('../private.json')))
    # Windows junction checks without creating any real symlink or directory.
    class Ancestor:
        def __init__(self,parent=None,reparse=False):self.parent=parent or self;self.reparse=reparse
        def is_relative_to(self,root):return True
        def exists(self):return True
        def is_symlink(self):return False
        def lstat(self):return SimpleNamespace(st_file_attributes=0x400 if self.reparse else 0)
    root=Ancestor();parent=Ancestor(root,True);leaf=Ancestor(parent)
    reject('planned_destination_reparse_ancestor_rejected_without_real_junction',lambda:p.reject_reparse(leaf,root))
    p.reject_reparse(Ancestor(root),root);mark('ordinary_planned_destination_ancestor_allowed',True)
    for name,payload in [
        ('secret_generated',lambda:p.write_new(None,('github_pat_'+'a'*30).encode())),
        ('both_ends_navigation_change',lambda:p.check_navigation_note(b'body',b'2026-10-08 body footer','nav')),
        ('oversized_navigation',lambda:p.check_navigation_note(b'body',b'body2026-10-08'+b'a'*65536,'nav')),
        ('state_under_v2_extra',lambda:p.bounded_extra('outputs/round117_v2/development/model/state.json')),
        ('private_prior_v2_extra',lambda:p.bounded_extra('outputs/round117_v2/development/prior/preparation.json')),
        ('per_row_price_object',lambda:p.policy('outputs/round117_v2/results.json',b'{"items":[{"row_index":9,"target":10}]}')),
        ('long_unlabelled_numeric_vector',lambda:p.check_aggregate({'arbitrary':[1.]*256})),
        ('JSON_duplicate_key',lambda:p.policy('outputs/round117_v2/results.json',b'{"passed":true,"passed":false}')),
    ]:reject(name,payload)
    # Sidecar names may not escape the directory or silently name a second file.
    with patch.object(p,'safe',side_effect=forbid):
        reject('sidecar_parent_traversal_rejected_before_read',lambda:p.check_sidecar('outputs/round117_v2/results.sha256',('a'*64+' ../other.json').encode()))
        reject('sidecar_multiple_records_rejected_before_read',lambda:p.check_sidecar('outputs/round117_v2/results.sha256',(('a'*64+'\n')*2).encode()))
    # Disposition guards in memory: gate failure and eligible-deferred are distinct.
    class Missing:
        def exists(self):return False
        def glob(self,pattern):return []
        def __truediv__(self,part):return self
    def gates(a):
        return {'august':{'exploratory_submission':{'components':{'x':a},'passed':a},
                'promotion':{'components':{'x':False},'adoption_research_gate_passed':False,'confirmation_fractional_improvement':0.}},
            'extra_origin':{'components':{'x':True},'passed':True},
            'combined_exploratory_submission':{'components':{'original_three_windows':a,'fixed_extra_origin':True},'passed':a},
            'combined_promotion':{'components':{'original_three_windows':False,'fixed_extra_origin':True},'passed':False}}
    with patch.object(p,'ROOT',Missing()),patch.object(p,'safe',return_value=Missing()):
        mark('failed_gate_no_release_only',p.check_disposition(117,{'no_release_reason':'fixed combined gate failed'},{'gates':gates(False)})=='combined_gate_failed_no_release')
        mark('eligible_gate_deferred_only',p.check_disposition(117,{'deferred_release_reason':'no full release authorized yet'},{'gates':gates(True)})=='eligible_full_release_deferred')
        reject('eligible_cannot_claim_no_release_failure',lambda:p.check_disposition(117,{'no_release_reason':'fixed gate failed'},{'gates':gates(True)}))
        reject('failed_cannot_claim_eligible',lambda:p.check_disposition(117,{'deferred_release_reason':'not completed yet'},{'gates':gates(False)}))
    class Existing(Missing):
        def exists(self):return True
    with patch.object(p,'ROOT',Missing()),patch.object(p,'safe',return_value=Existing()):
        reject('full_release_requires_new_reviewed_schema',lambda:p.check_disposition(117,{'deferred_release_reason':'no full release authorized yet'},{'gates':gates(True)}))
    c.need(c.sha(SOURCE)==SOURCE_SHA,'Source changed during pure independent review')
    return {'status':'passed_synthetic_source_checks_not_formal_package_check','source':c.record(SOURCE),
        'review_tool':c.record(__file__),'author_suite_reexecuted':author,'independent_checks':checks,
        'independent_check_count':len(checks),'new_model_fits':0,'real_meta_fits':0,'GPU_work':False,
        'formal_package_check_executed':False,'actual_repository_members_hashed':False,'production_repository_written':False,
        'copies_Git_network_or_real_array_model_prices_read':False,'elapsed_seconds':time.monotonic()-tick}


if __name__=='__main__':
    result=run();c.write('outputs/research_continue_20261008/package_four_origins_synthetic_review.json',result)
    print(json.dumps({'status':result['status'],'independent_checks':result['independent_check_count']},ensure_ascii=False))
