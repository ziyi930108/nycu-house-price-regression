"""Root's real fit-only four-origin context replay, before any R114 fitting."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c
import round114_research as driver


def main():
    tick=time.monotonic();source=driver.sources();inputs={}
    specifications=driver.august_specifications(inputs)
    extra=driver.offset_comp.specification()
    driver.merge_hashes(inputs,extra['input_sha256'])
    specifications[driver.OFFSET]={'composition':extra,'prepared':extra['prior_npz']}
    c.verify_hashes(inputs);protected=driver.protected();rows={}
    for origin in driver.ORIGINS:
        start=time.monotonic();loaded=driver.load_context({'origins':specifications},origin)
        driver.verify_context(loaded,origin)
        rows[origin]={'fit_rows':len(loaded['fit_X']),'query_rows':len(loaded['query_X']),
            'prepared_array_identities':loaded['array_identities'],'context':loaded['before'],
            'original_v22_v24_replayed_exact':True,'elapsed_seconds':time.monotonic()-start}
        print(origin,round(rows[origin]['elapsed_seconds'],3),flush=True)
        del loaded
    c.need(source==driver.sources() and protected==driver.protected(),'Source/protected changed during CPU replay')
    c.verify_hashes(inputs)
    c.write('outputs/research_round114_20261008/root_context_preflight.json',{
        'status':'passed','source_sha256':{**source,driver.PLAN:c.sha(driver.PLAN)},'input_sha256':inputs,
        'origin_order':list(driver.ORIGINS),'origins':rows,'elapsed_seconds':time.monotonic()-tick,
        'query_prices_read':False,'raw_price_pickle_deserialized':False,'saved_fit_labels_read':True,
        'new_fits':0,'GPU_calls':0,'native_model_inference_calls':0,'new_CSV':False,
        'protected_unchanged':True,'review_tool':c.record(__file__),
        'scope':'Real saved fit-only contexts, arrays, v22/v24 and transform chains; independent source review remains required.'})


if __name__=='__main__':main()
