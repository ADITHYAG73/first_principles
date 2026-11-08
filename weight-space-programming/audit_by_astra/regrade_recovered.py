"""CPU-only audit. Executes only seven inspected pure scoring functions from Opus worker AST."""
import ast,json,re,math,hashlib,datetime
from pathlib import Path
from collections import defaultdict
from importlib.metadata import version
ROOT=Path(__file__).resolve().parent
P=ROOT/'recovered-heldout'
worker=Path('/Users/adithyagiridharan/Desktop/PythonProjects/claude/remote/worker.py')
source=worker.read_text();tree=ast.parse(source)
names={'last_boxed','grade','repetition_stats','finalize_gen','summarize','paired'}
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in selected}==names
exec(compile(ast.Module(body=selected,type_ignores=[]),str(worker),'exec'))
gold=json.loads((P/'reconstructed-selection-private.json').read_text())
items={k:{'gold':v['answer']} for k,v in gold['gold'].items()}
results={}; summaries={}
for m in ['original','edited']:
 rows=[json.loads(l) for l in (P/f'gen_{m}_gpu0.jsonl').read_text().splitlines()]
 graded=[]
 for i,r in enumerate(rows):
  graded.extend(finalize_gen([r],items))
  if i%64==63:print(m,i+1,'graded',flush=True)
 results[m]=graded;summaries[m]=summarize(graded)
 with (P/f'regraded-{m}.jsonl').open('w') as f:
  for r in graded:f.write(json.dumps({k:v for k,v in r.items() if k not in {'text','prompt_token_ids','token_ids'}})+'\n')
summaries['paired']=paired(results['original'],results['edited'])
expected=json.loads((P/'status.json').read_text())['summary']
orig={(r['id'],r['sample']):r for r in results['original']};edit={(r['id'],r['sample']):r for r in results['edited']}
report={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'math_verify_version':version('math-verify'),'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'procedure':'Regraded recovered texts using extracted original scoring functions; no inference; reconstructed gold from local official source caches; prompt text matched every problem. Original installed math-verify version unverified.','recomputed':summaries,'reported':expected,'exact_summary_match':json.dumps(summaries,sort_keys=True)==json.dumps(expected,sort_keys=True),'paired_seeds_match':all(orig[k]['seed']==edit[k]['seed'] for k in orig),'correct_but_truncated':{m:[r['rid'] for r in rows if r['correct'] and r['category']=='cut_off'] for m,rows in results.items()},'source_meta':gold['meta']}
(P/'regrading-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in {'reported','source_meta'}},indent=2))
