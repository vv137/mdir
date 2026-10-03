"""The backups that `mdir run` keeps of the outputs of an earlier run
(D149), `#<name>.<n>#` in the directory of the output.

    backups.py list DIRECTORY        prints the names of the backups, sorted
    backups.py make PATH COUNT       makes PATH and COUNT backups of it"""
import os
import sys

if sys.argv[1] == 'list':
    names = sorted(name for name in os.listdir(sys.argv[2])
                   if name.startswith('#') and name.endswith('#'))
    print('backups: %s' % (' '.join(names) if names else 'none'))
else:
    path, count = sys.argv[2], int(sys.argv[3])
    directory, name = os.path.split(path)
    open(path, 'w').close()
    for n in range(1, count + 1):
        open(os.path.join(directory, '#%s.%d#' % (name, n)), 'w').close()
