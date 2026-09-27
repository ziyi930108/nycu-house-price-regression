"""Read-only artifact checks for the completed fourth round, plus a JSON receipt."""
from pathlib import Path
from datetime import datetime, timezone
import json, re
from urllib.parse import unquote
import numpy as np
import pandas as pd
from round4_common import ROOT, OUT, plan, read_locked, verify_hashes, read
from prepare_data import sha256


def main():
    config=plan()
    frozen=read_locked(OUT/'frozen_candidate.json')
    accepted=read_locked(OUT/'accepted_candidate.json')
    result=read_locked(OUT/'evaluation/development/results.json')
    verify_hashes(frozen['source_sha256'])
    verify_hashes(frozen['development_artifact_sha256'])
    verify_hashes(accepted['evidence_sha256'])
    assert accepted['accepted'] is False and accepted['deployment_decision']=='retain_v2'
    assert frozen['selected_candidate']=='v2_control'
    assert not result['decision']['qualified_candidates']
    for stage in ('confirmation','auxiliary','final'):
        assert not (OUT/'models'/stage).exists(), 'Unexpected later-stage model'
        assert not (OUT/'evaluation'/stage/'results.json').exists(), 'Unexpected later-stage score'
    assert sha256(ROOT/'runs/round3_experiments/snapshot_manifest.json')=='96cee2c7977d9317f70e5e64b41cbf057db8fad529019a2e76fb37d1606a202f'
    submission=ROOT/'outputs/submission_v2.csv'
    assert sha256(submission)=='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
    csv=pd.read_csv(submission)
    sample_paths=[ROOT/p for p in config['data_sha256'] if 'sample-submission' in p and p.endswith('.csv')]
    assert len(sample_paths)==1
    sample=pd.read_csv(sample_paths[0])
    assert list(csv.columns)==['編號','總價元'] and len(csv)==47073
    assert list(sample.columns)==list(csv.columns)
    assert np.array_equal(csv['編號'].to_numpy(),sample['編號'].to_numpy())
    assert csv['編號'].is_unique and np.isfinite(csv['總價元']).all() and csv['總價元'].gt(0).all()
    summary=read(ROOT/'reports/round4_results.json')
    assert summary['status']=='retain_v2' and summary['private_mae'] is None
    assert summary['stages']['development']['metrics']==result['metrics']
    for stage in ('confirmation','auxiliary'):
        assert summary['stages'][stage]['status']=='not_required' and summary['stages'][stage]['metrics'] is None
    docs=sorted(set((ROOT/'reports').glob('*round4*.md'))|{ROOT/'README_round4.md'})
    destination=ROOT/'reports/round4_final_checks.json'
    checked=[]
    broken=[]
    for doc in docs:
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z]+://',target) or target.startswith('#'):
                continue
            target=unquote(target.split('#',1)[0].strip('<>'))
            actual=(doc.parent/target).resolve()
            if not actual.exists() and actual!=destination:
                broken.append({'file':str(doc.relative_to(ROOT)),'link':target})
            checked.append({'file':str(doc.relative_to(ROOT)),'link':target})
    assert not broken, broken
    payload={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':sha256(OUT/'candidate_plan.json'),'freeze_sha256':sha256(OUT/'frozen_candidate.json'),
        'acceptance_sha256':sha256(OUT/'accepted_candidate.json'),'accepted':False,'recommended_submission':'outputs/submission_v2.csv',
        'submission_sha256':sha256(submission),'submission_rows':len(csv),'submission_id_order_matches_sample':True,
        'submission_finite_positive':True,'original_data_and_frozen_sources_unchanged':True,
        'round3_snapshot_manifest_unchanged':True,'development_metrics_match_final_report':True,
        'new_later_stage_models_or_scores_present':False,'new_submission_created':False,
        'historically_untouched':False,'public_score_used_to_tune':False,'private_mae':None,
        'local_markdown_link_count':len(checked),'broken_links':broken,
        'checked_docs_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in docs},
        'script_sha256':sha256(Path(__file__))}
    destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','submission_rows':len(csv),'checked_links':len(checked)},ensure_ascii=False))

if __name__=='__main__':
    main()
