"""Derive a new, auditable batch entry without editing old experiment sources."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
parent = ROOT / 'src/round109_110_research.py'
target = ROOT / 'src/round111_112_research.py'
text = parent.read_text(encoding='utf-8-sig')
for old, new in (
    ('round109_110', 'round111_112'),
    ('research_round111_112_20261008', 'research_round111_112_20261008'),
    ('round109_tail_parameter_average_train.py', 'round111_residual_prior_train.py'),
    ('round110_target_borders_unit.py', 'round112_ordered_unit.py'),
    ('round110_target_borders_unit as helper', 'round112_ordered_unit as helper'),
    ('R109_average_selfcheck_execution.json', 'R111_residual_selfcheck_execution.json'),
    ('R110_helper_selfcheck_execution.json', 'R112_helper_selfcheck_execution.json'),
    ('round109', 'round111'), ('round110', 'round112'),
    ('R109', 'R111'), ('R110', 'R112'),
    ('rec109', 'rec111'), ('rec110', 'rec112'),
    ('(109,110)', '(111,112)'), ('n==109', 'n==111'),
):
    text = text.replace(old, new)
start = text.index("    epoch30=np.load(folder/'model/predictions_epoch30_twd.npy'")
end = text.index("    folder=c.ROOT/'outputs/round112'/s", start)
text = text[:start] + """    c.write(folder/'stage.json',{'stage':s,'layers':rec111,'model_metadata':c.record(folder/'model/training.json'),
                             'sealed_models':[c.record(folder/'model/model.pt'),c.record(folder/'model/run.json')],
                             'fit_rows':nf,'query_rows':nq,'fit_seconds':meta['fit_seconds'],'query_labels_available':False})
""" + text[end:]
start = text.index('            if n==111:')
end = text.index("            c.write(folder/'results.json'", start)
text = text[:start] + text[end:]
text = text.replace("'sealed_models':[c.record(folder/'model/model.cbm')],",
                    "'sealed_models':[c.record(folder/'model/model.cbm'),c.record(folder/'model/state.json'),c.record(folder/'model/native_model_summary.json')],")
# All models are directly bound before evaluation, unlike the original R107 seal.
text = text.replace('"""Two preregistered serial experiments; fit/query state and scoring separated."""',
                    '"""R111 ResNet prior and R112 Ordered unit; six fixed serial fits, then scoring."""')
if any(word in text for word in ('round109', 'round110', 'epoch30', 'tail_parameter_average', 'target_borders_unit')):
    raise AssertionError('Unexpected previous-experiment semantics remain')
compile(text, str(target), 'exec')
with target.open('x', encoding='utf8', newline='\n') as handle:
    handle.write(text)
receipt = {
    'parent': str(parent.relative_to(ROOT)).replace('\\', '/'),
    'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'target': str(target.relative_to(ROOT)).replace('\\', '/'),
    'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'status': 'source_prepared_no_fit', 'old_source_modified': False,
    'CPU_compile_passed': True, 'new_model_fits': 0,
}
with (Path(__file__).parent / 'source_preparation.json').open('x', encoding='utf8') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps(receipt), flush=True)
