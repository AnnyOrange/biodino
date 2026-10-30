"""Available RAM constrained by all cgroup memory ancestors (v1 and v2)."""
from pathlib import Path


def memory(proc=Path('/proc')):
    info = {k: int(v.split()[0]) * 1024 for k, v in
            (line.split(':', 1) for line in (proc / 'meminfo').read_text().splitlines())}
    total, available = info['MemTotal'], info['MemAvailable']
    memberships = []
    for line in (proc / 'self/cgroup').read_text().splitlines():
        _, controllers, path = line.split(':', 2)
        if 'memory' in controllers.split(','):
            memberships.append(('cgroup', path))
        elif not controllers:
            memberships.append(('cgroup2', path))
    seen = set()
    for line in (proc / 'self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        fields, filesystem = before.split(), after.split()
        kind = filesystem[0]
        if kind not in ('cgroup', 'cgroup2'):
            continue
        if kind == 'cgroup' and 'memory' not in filesystem[2].split(','):
            continue
        # mountinfo encodes spaces, tabs, newlines and backslashes with octal.
        def unescape(value):
            for code, char in [('040', ' '), ('011', '\t'), ('012', '\n'), ('134', '\\')]:
                value = value.replace('\\' + code, char)
            return value
        root, mount = Path(unescape(fields[3])), Path(unescape(fields[4]))
        for member_kind, member in memberships:
            if member_kind != kind:
                continue
            try:
                relative = Path(member).relative_to(root)
            except ValueError:
                continue
            directory = mount / relative
            while True:
                if directory not in seen:
                    seen.add(directory)
                    limit_file = directory / ('memory.max' if kind == 'cgroup2' else 'memory.limit_in_bytes')
                    if limit_file.exists():
                        raw = limit_file.read_text().strip()
                        if raw != 'max':
                            limit = int(raw)
                            # Kernel v1 uses a near-LONG_MAX sentinel for unlimited.
                            if limit < (1 << 60):
                                used = int((directory / ('memory.current' if kind == 'cgroup2' else 'memory.usage_in_bytes')).read_text())
                                stats = dict(x.split() for x in (directory / 'memory.stat').read_text().splitlines())
                                inactive = int(stats.get('inactive_file' if kind == 'cgroup2' else 'total_inactive_file', '0'))
                                # Only inactive file cache is considered reclaimable;
                                # never count swap or all cached pages as free RAM.
                                working = max(0, used - min(used, inactive))
                                total = min(total, limit)
                                available = min(available, max(0, limit - working))
                if directory == mount:
                    break
                directory = directory.parent
    return total, available
