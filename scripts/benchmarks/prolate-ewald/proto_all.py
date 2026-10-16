"""GPU job: proto_job.py for several systems in one job.
proto_all.py DEVICE_TAG LAYOUTS PRECS SYSTEM..."""
import os, subprocess, sys

D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
tag, layouts, precs = sys.argv[1:4]
fail = 0
for s in sys.argv[4:]:
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_job.py'), f'{D}cases/cases-{s}.json', f'{D}proto2-{s}-{tag}.jsonl', layouts, precs])
    fail += r.returncode != 0
sys.exit(1 if fail else 0)
