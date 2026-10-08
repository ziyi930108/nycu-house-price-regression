"""Independent bounded R117 recovery source/lineage review; CPU only, no fit."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,time,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,m):
 if not x:raise ValueError(m)
def main():
 start=time.perf_counter();old=ROOT/'src/round117_research.py';new=ROOT/'src/round117_research_v2.py'
 before=sha(new);need(before=='b2558a7ce7ae11814c624f9058598d931c4c95b0237776f266eff6ceb50b0a29','Reviewed source changed')
 parse=lambda p:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef)}
 a,b=parse(old),parse(new)
 science=['fixed_folds','ratio_bias','check_changed_arrays','load_context','prior_records','check_prior_evidence','prepare','ratio_total','model_records','validate','stage','train','combined_gates','scoring_preflight']
 need(all(a[x]==b[x] for x in science),'Scientific/stage AST differs')
 changed=sorted(k for k in a if a[k]!=b.get(k));need(changed==sorted(['sources','specifications','freeze','request','check_metadata','evaluate','self_check']),'Unexpected original function changed')
 spec=importlib.util.spec_from_file_location('independent_r117_recovery',new);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 sources=m.sources();actual=m.actual_trainer_lineage();need(len(sources)==51 and len(actual)==4,'51-source/four-parent contract')
 guards=[]
 for k in actual:
  for mode in ('missing','wrong_SHA'):
   bad=dict(sources)
   if mode=='missing':bad.pop(k)
   else:bad[k]='0'*64
   try:m.check_source_lineage(bad)
   except ValueError:guards.append(mode+':'+k)
   else:raise ValueError('Actual lineage guard accepted '+mode+':'+k)
 text_trainer=(ROOT/m.TRAINER).read_text(encoding='utf-8');text_parent=(ROOT/'src/round63_recency_weighted_prior_train.py').read_text(encoding='utf-8')
 mutations=[('trainer_import',m.TRAINER,text_trainer.replace('import round63_recency_weighted_prior_train as baseline','import another_trainer as baseline')),
 ('trainer_tuple',m.TRAINER,text_trainer.replace('baseline.PARENT_TRAINER','Path(prior.__file__)')),
 ('trainer_parent',m.TRAINER,text_trainer.replace('PARENT_TRAINER = Path(baseline.__file__)','PARENT_TRAINER = Path(prior.__file__)')),
 ('parent_filename','src/round63_recency_weighted_prior_train.py',text_parent.replace('round52_nn_train.py','round47_nn_train.py')),
 ('parent_expression','src/round63_recency_weighted_prior_train.py',text_parent.replace('.with_name(','.with_suffix('))]
 original_safe=m.c.safe;ast_guards=[]
 class SyntheticSource:
  def __init__(self,path,text):self.path,self.text=path,text
  def read_text(self,*args,**kwargs):return self.text
  def with_name(self,name):return self.path.with_name(name)
 try:
  for label,name,text in mutations:
   need(text!=(text_trainer if name==m.TRAINER else text_parent),'Mutation did not change actual source')
   def safe(path,name=name,text=text):
    return SyntheticSource(ROOT/name,text) if str(path)==name else original_safe(path)
   m.c.safe=safe
   try:m.actual_trainer_lineage()
   except ValueError:ast_guards.append(label)
   else:raise ValueError('Actual trainer AST guard accepted '+label)
 finally:m.c.safe=original_safe
 need(m.sources()==sources and m.check_source_lineage(sources)==actual and sha(new)==before,'Source changed during review')
 author=ROOT/'outputs/research_round117_v2_20261008/driver_author_review.json';av=json.loads(author.read_bytes())
 merged={**sources,m.PLAN:sha(ROOT/m.PLAN),m.RECOVERY_PLAN:sha(ROOT/m.RECOVERY_PLAN)}
 need(av['source_sha256']==merged and av['exit_code']==0,'Author exact source/plan receipt differs')
 records={k:{'path':k,'sha256':sha(ROOT/k)} for k in ['src/round117_research.py','src/round117_research_v2.py',m.PLAN,m.RECOVERY_PLAN,m.TRAINER,'src/round63_recency_weighted_prior_train.py','src/round47_nn_train.py','src/round52_nn_train.py']}
 return {'status':'passed_bounded_independent_source_audit','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':{'path':str(new.relative_to(ROOT)).replace('\\','/'),'sha256':before},'review_tool':{'path':str(Path(__file__).relative_to(ROOT)).replace('\\','/'),'sha256':sha(Path(__file__))},'source_sha256':merged,'readset':records,'original_functions_AST_changed':changed,'scientific_stage_functions_AST_exact':science,'independent_dynamic_missing_wrong_SHA_guards':guards,'independent_actual_AST_mutation_guards':ast_guards,'independent_dynamic_guard_count':8,'independent_AST_mutation_guard_count':5,'elapsed_seconds':time.perf_counter()-start,'new_model_fits':0,'new_prior_builds':0,'new_prepared_arrays':0,'GPU_work':False,'query_prices_or_new_scores_read':False,'old_models_or_priors_loaded_or_reused':False,'full_independent_preflight_passed':False,'limits':['This is source/actual lineage guard review only. Formal CPU receipts, exact current input hashes and root four-context replay remain required before full preflight.','Author source review is retained separately; no true prior build, NN training, query labels, scoring or CSV.']}
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
