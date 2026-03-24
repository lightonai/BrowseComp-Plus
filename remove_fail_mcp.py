import json, glob, os

files = glob.glob('runs/pylate_gpt5-reason-mcp/run_*.json')
removed = 0
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    for r in d.get('result', []):
        if r.get('type') == 'tool_call' and r.get('output') is None:
            print(f"Removing qid={d.get('query_id')}  {f}")
            os.remove(f)
            removed += 1
            break
print(f'\nRemoved {removed} files')
