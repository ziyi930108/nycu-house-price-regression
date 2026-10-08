"""Bounded independent source/guards/old-manifest review; never real apply."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
SOURCE='src/package_research_continue_20261008_v2.py'
SHA='a9071908de685677bcbd1bc726d983c6fe5052f953afd1ea23bbbce587858b54'
tick=time.monotonic();c.need(c.sha(SOURCE)==SHA,'Final v2 packager source changed')
spec=importlib.util.spec_from_file_location('independent_packager',c.safe(SOURCE))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
checks=[]
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError):checks.append(name)
    else:raise AssertionError('Guard accepted '+name)

# Re-execute author's pure memory suite, then independent cases below. This
# is separated from novel adversarial checks in the final receipt.
author_suite=p.self_check()
c.need(author_suite['status']=='passed' and author_suite['check_count']==76,'All author CPU cases')
actual=p.package_files();template,protected=p.protect_existing(actual)
c.need(len(template['files'])==3017 and len(actual)==3018 and len(protected)==3014,'Explicit original manifest counts')
checks.append('all_original_3017_members_full_bytes_and_SHA_verified')
c.need(p.check_git_base(item['path'] for item in template['files'])['HEAD']==p.BASE_HEAD,'Exact base/index membership')
checks.append('original_HEAD_and_3018_index_names_verified_without_Git')
victim=next(name for name in protected if name.startswith('evidence/'))
bad=copy.deepcopy(actual);bad[victim]['bytes']+=1
reject('nonproduction_old_evidence_bytecount_mutation',lambda:p.protect_existing(bad))
bad=copy.deepcopy(actual);bad[victim]['sha256']='0'*64
reject('nonproduction_old_evidence_SHA_mutation',lambda:p.protect_existing(bad))
bad=copy.deepcopy(actual);bad.pop(victim)
reject('nonproduction_old_evidence_membership_mutation',lambda:p.protect_existing(bad))
for label,call in (
    ('generated_credential_write_rejected_before_FS',lambda:p.write_new(None,('sk-'+'A'*30).encode())),
    ('disposition_credential_rejected',lambda:p.parse_dispositions([],['111=token github_pat_'+'A'*30,'112=failed fixed gates'])),
    ('JSON_ambiguous_key_rejected',lambda:p.policy('outputs/round113/x.json',b'{"gate":true,"gate":false}')),
    ('per_ID_price_record_rejected',lambda:p.policy('outputs/round113/x.json',b'{"nested":[{"sample_id":"s","price_twd":100}]}')),
    ('embedded_loss_vector_rejected',lambda:p.check_aggregate({'metric':[float(n) for n in range(256)]})),
    ('manifest_private_ancestor_rejected',lambda:p.relative('evidence/private/secret.json')),
    ('manifest_runs_ancestor_rejected',lambda:p.relative('evidence/runs/metadata.json')),
    ('extra_state_file_rejected',lambda:p.policy('outputs/round113/R107/state.json')),
    ('navigation_internal_body_change_rejected',lambda:p.check_navigation_note(b'abc\n',b'2026-10-08\nchanged\n','nav')),
    ('navigation_both_ends_change_rejected',lambda:p.check_navigation_note(b'abc\n',b'2026-10-08\nabc\nfooter','nav')),
):reject(label,call)
# A record with one correct field cannot pass as a reviewed artifact.
with patch.object(p,'reference',return_value={'path':'outputs/round113/receipt.json','sha256':'a'*64}):
    reject('extra_reference_wrong_digest',lambda:p.check_record({'path':'outputs/round113/receipt.json','sha256':'b'*64}))
    reject('extra_reference_ignored_fields',lambda:p.check_record({'path':'outputs/round113/receipt.json','sha256':'a'*64,'passed':True}))
c.need(c.sha(SOURCE)==SHA,'Source changed during independent review')
receipt={'status':'passed_source_review_not_formal_package_check','created_at_utc':c.now(),
         'reviewed_source':c.record(SOURCE),'review_tool':c.record(__file__),
         'author_review':c.record('outputs/research_continue_20261008/direction_review_a/package_v2_author_review.json'),
         'author_selfcheck':c.record('outputs/research_continue_20261008/direction_review_a/package_v2_author_selfcheck.json'),
         'author_pure_memory_suite_independently_reexecuted':author_suite,
         'independent_checks':checks,'independent_check_count':len(checks),
         'actual_baseline':{'base_HEAD':p.BASE_HEAD,'old_manifest_sha256':p.BASE_MANIFEST_SHA256,
                            'old_manifest_members':3017,'index_members':3018,'protected_members':3014,
                            'historical_source_copies':379,'production_sources':13},
         'original_source_attempt1':c.record('src/package_research_continue_20261008.py'),
         'original_source_attempt1_blocker':'R113 review allowed only status/source/gate/results/plan without mandatory independent identity/scope/manual/artifact bindings.',
         'fixed_in_reviewed_v2':['Mandatory exact R113 request/seal/execution/result/gate/plan/tool identities.',
                                'Every 107/111/112 stage, exact complete sealed model list, full metadata and manual gate verdict must match.',
                                'Protected file bytes checked; no-fit/GPU/CSV and unknown scores/preserved v22 required.',
                                'Only a bare digest or digest plus exact same-directory JSON filename is allowed as a single sidecar.'],
         'blocking_findings_remaining':[],
         'scope_limits':['Formal --check not run: R113 completion/postfit receipts are required and were not read early.',
                         'No real apply/copy/refresh/Git/network/fit/GPU/query-price access.',
                         'Actual native models/arrays/CSV are opaque hash references only in package; no content copied.',
                         'Reparse-ancestor checks are source-reviewed, not tested by creating real Windows junctions.',
                         'Git index names/checksum/base identity are checked; staged blob content is not separately compared.',
                         'Partial copy is intentionally preserved and blocked from silent rerun, not atomic/rollback.'],
         'formal_package_check_performed':False,'deliverables_writes':False,'file_copies':0,'git_commands':0,
         'new_model_fits':0,'GPU_work':False,'query_y_read':False,'network':False,
         'review_elapsed_seconds':time.monotonic()-tick}
c.write('outputs/research_continue_20261008/package_independent_review.json',receipt)
print(json.dumps({'status':receipt['status'],'independent_checks':len(checks),
                  'review':c.record('outputs/research_continue_20261008/package_independent_review.json')},ensure_ascii=False))
