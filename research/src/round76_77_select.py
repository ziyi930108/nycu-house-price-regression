"""After verified local CSV release, preserve the previous selector and record R76/R77 preference."""
from pathlib import Path
import argparse, hashlib, json, os
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round76_77/selection'
SELECTOR=ROOT/'outputs/current_private_candidate.json'
OLD_SHA='f5af20180e2ee77f12eb90058930e139c0256f29b0e943581ee7c44d293ee900'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def locked(p):
    assert sha(p)==p.with_suffix('.sha256').read_text(encoding='ascii').strip()
    return json.loads(p.read_text(encoding='utf8'))
def write(p,obj):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')
def run():
    assert sha(SELECTOR)==OLD_SHA and not OUT.exists()
    release_path=ROOT/'outputs/round76_77/final_export/release.json'
    release=locked(release_path)
    assert locked(release_path.parent/'execution.json')['exit_code']==0
    joint=locked(ROOT/'outputs/round76_77/final/trial_release.json')
    assert joint['status']=='final_vectors_and_export_payloads_ready_no_CSV'
    assert all(sha(ROOT/n)==h for n,h in joint['protected_before'].items())
    assert not SELECTOR.with_suffix('.sha256').exists(),'Unexpected mutable selector receipt'
    choices=[]
    for r,v in ((76,20),(77,21)):
        author=locked(ROOT/f'outputs/round76_77/csv_authoring/round{r}/validation.json')
        path=ROOT/f'outputs/submission_v{v}_trial.csv'
        assert sha(path)==author['csv_sha256'] and author['all_saved_CSV_cells_exact'] and author['all_typed_cells_exact']
        proof_path=ROOT/f'outputs/round{r}/final/trial_release.json';proof=locked(proof_path)
        choices.append({'round':r,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
            'rows':47073,'final_proof':proof_path.relative_to(ROOT).as_posix(),'final_proof_sha256':sha(proof_path),
            'public_mae':None,'private_mae':None})
    before=SELECTOR.read_bytes();previous=json.loads(before.decode('utf8'));OUT.mkdir()
    with (OUT/'previous_selector.json').open('xb') as f:f.write(before)
    with (OUT/'previous_selector.sha256').open('x',encoding='ascii') as f:f.write(OLD_SHA+'\n')
    updated=dict(previous);completed=datetime.now(timezone.utc).isoformat()
    updated.update(updated_at_utc=completed,scope='local_recommendation_only',
        private_priority=choices[0]['path'],sha256=choices[0]['sha256'],backup=choices[1]['path'],
        backup_sha256=choices[1]['sha256'],previous_candidate=previous['private_priority'],
        known_public_best_filename='outputs/submission_v18_trial.csv',known_public_best_mae=2388386.4613,
        public_mae=None,private_mae=None,kaggle_selection_changed=False,final_version_chosen=False,
        GitHub_publication_deferred=True,
        previous_selection_metadata={key:previous.get(key) for key in
            ('latest_completed_research','current_priority_release','backup_role')},
        latest_completed_research={'round':77,'status':'released_two_local_trial_candidates',
            'decision':'local_R76_priority_with_R77_complementary_alternative',
            'decision_record':'outputs/round77/decision.json',
            'decision_sha256':sha(ROOT/'outputs/round77/decision.json'),
            'report':'reports/round77_report.md','new_final_NN_fits':1,
            'completed_at_utc':completed,'final_version_chosen':False},
        current_priority_release={'round':76,'candidate':'R76',
            'decision':'limited_local_private_preference_after_verified_CSV_release',
            'submission':choices[0]['path'],'submission_sha256':choices[0]['sha256'],
            'final_proof':choices[0]['final_proof'],'final_proof_sha256':choices[0]['final_proof_sha256'],
            'CSV_release':release_path.relative_to(ROOT).as_posix(),'CSV_release_sha256':sha(release_path),
            'backup':choices[1]['path'],'backup_sha256':choices[1]['sha256'],
            'public_mae':None,'private_mae':None,'uploaded':False,'kaggle_selection_changed':False,
            'final_version_chosen':False,'completed_at_utc':completed},
        backup_role='R77/v21 is a complementary cross-period alternative to R76/v20; its Public and Private MAEs are unknown.',
        pending_research={'round':77,'status':'completed_two_local_trial_CSVs_released','new_final_NN_fits':1},
        reason='Tentative R76 priority for lower newest-period overall and test-composition MAE; R77 alternative improves both earlier periods. R77-minus-R76 newest-period intervals cross zero. Reused windows, single seed and high-price concentration limit generalization; Private remains unknown.',
        candidates=choices,final_release=release_path.relative_to(ROOT).as_posix(),final_release_sha256=sha(release_path),
        previous_selector_backup='outputs/round76_77/selection/previous_selector.json',
        limitations=['No fresh blind holdout or correction for many rounds of adaptive selection.',
            'R76 newest-period gain is substantially concentrated in true prices at least 50m TWD.',
            'R77 newest-period new-project and >=100m groups worsen relative to R76.',
            'Public and Private for v20/v21 are unknown; no automatic Kaggle upload or final selection.'])
    updated['evidence']=list(dict.fromkeys(previous.get('evidence',[])+['reports/round76_report.md','reports/round77_report.md',
        'outputs/round77/decision.json','reports/round76_77_final_plan.md',release_path.relative_to(ROOT).as_posix()]))
    staged=OUT/'new_selector.json';write(staged,updated)
    assert sha(SELECTOR)==OLD_SHA
    replacement=ROOT/'outputs/current_private_candidate.next.json'
    with replacement.open('xb') as f:f.write(staged.read_bytes())
    os.replace(replacement,SELECTOR)
    assert sha(SELECTOR)==sha(staged)
    write(OUT/'decision.json',{'status':'local_recommendation_updated','previous_selector_sha256':OLD_SHA,
        'new_selector_sha256':sha(SELECTOR),'source_sha256':sha(Path(__file__)),'candidates':choices,
        'release_sha256':sha(release_path),'final_version_chosen':False,'private_mae':None,'uploaded':False})
    print(json.dumps({'status':'local_recommendation_updated','priority':choices[0]['path'],'alternative':choices[1]['path']}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true',required=True);p.parse_args();run()
