"""Compatibility entrypoint: external queue replacement is permanently disabled.

SIGSTOP-based replacement can strand workers if its controller dies. Updated
workers reload themselves only between tasks, retaining their PID and slot lock.
Old workers are allowed to finish; this entrypoint never signals or spawns them.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resident")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"status": "disabled_external_rollout", "host": args.resident,
                      "reason": "Workers reload cooperatively at idle boundaries"}))


if __name__ == "__main__":
    main()
