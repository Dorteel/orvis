#!/usr/bin/env python3
import time, subprocess, hashlib, os, json

WATCH_PATH = "."
STATE_FILE = ".watch_hash.json"
INTERVAL = 60  # seconds

def hash_dir(path):
    h = hashlib.sha256()
    for root, _, files in os.walk(path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            h.update(fp.encode())
            h.update(str(os.path.getmtime(fp)).encode())
    return h.hexdigest()

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# load previous state
if os.path.exists(STATE_FILE):
    old_hash = json.load(open(STATE_FILE))
else:
    old_hash = ""

while True:
    new_hash = hash_dir(WATCH_PATH)
    if new_hash != old_hash:
        print("Change detected, committing...")
        old_hash = new_hash
        json.dump(old_hash, open(STATE_FILE, "w"))

        try:
            run("git add .")
            run(f'git commit -m "Auto: results update"')
            run("git pull --rebase")
            run("git push")
        except Exception:
            print("Git push failed, continuing...")
    time.sleep(INTERVAL)
