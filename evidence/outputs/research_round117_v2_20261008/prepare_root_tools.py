"""Build new coordinator tools from preserved root-only first-attempt templates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import round100_contract as c

OLD = 'outputs/research_round117_20261008/'
NEW = 'outputs/research_round117_v2_20261008/'
def change(text, old, new, count=1):
    c.need(text.count(old) == count, 'Template occurrence count changed: ' + old)
    return text.replace(old, new)

launch = c.safe(OLD + 'launch.py').read_text(encoding='utf-8-sig')
launch = change(launch, 'outputs/research_round117_20261008', 'outputs/research_round117_v2_20261008')
launch = change(launch, "DRIVER = 'src/round117_research.py'", "DRIVER = 'src/round117_research_v2.py'")
replay = c.safe(OLD + 'replay_preflight.py').read_text(encoding='utf-8-sig')
replay = change(replay, 'import round117_research as driver', 'import round117_research_v2 as driver')
replay = change(replay, 'outputs/research_round117_20261008/root_context_preflight.json',
                        'outputs/research_round117_v2_20261008/root_context_preflight.json')
replay = change(replay, "{**sources, driver.PLAN: c.sha(driver.PLAN)}",
                        "{**sources, driver.PLAN: c.sha(driver.PLAN), driver.RECOVERY_PLAN: c.sha(driver.RECOVERY_PLAN)}")
replay = change(replay, "'R115_postfit': spec['R115_postfit'], 'saved_fit_labels_read': True,",
                        "'R115_postfit': spec['R115_postfit'], 'engineering_recovery': spec['engineering_recovery'], 'saved_fit_labels_read': True,")
for name, source in (('launch.py', launch), ('replay_preflight.py', replay)):
    compile(source, name, 'exec')
    with c.safe(NEW + name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(source)
c.write(NEW + 'root_tools_binding.json', {
    'status': 'new_recovery_tools_prepared_not_executed',
    'builder': c.record(__file__),
    'original_templates': {p: c.sha(p) for p in (OLD+'launch.py', OLD+'replay_preflight.py')},
    'new_tools': {p: c.sha(p) for p in (NEW+'launch.py', NEW+'replay_preflight.py')},
    'science_driver': c.record('src/round117_research_v2.py'),
    'new_fits': 0, 'query_y_read': False})
print('New recovery coordinator templates prepared; original tools unchanged')
