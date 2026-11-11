"""Static checks only: does not import or execute archived clinical scripts."""
from pathlib import Path
import ast,json,hashlib
root=Path(__file__).resolve().parents[1]
files=list((root/'analysis_source').rglob('*.py'))+list((root/'src').rglob('*.py'))+list((root/'scripts').glob('*.py'))
for file in files:ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
items=json.loads((root/'provenance/source_inventory.json').read_text(encoding='utf-8'))
for item in items:
    assert hashlib.sha256((root/item['path']).read_text(encoding='utf-8').encode('utf-8')).hexdigest()==item['release_sha256'],item['path']
print(json.dumps({'python_files_parsed':len(files),'source_hashes_verified':len(items),'clinical_pipeline_run':False}))
