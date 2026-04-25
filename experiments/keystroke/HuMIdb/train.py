"""Legacy HuMI entrypoint.

The cleaned keystroke path currently supports AaltoDB and HMOGDB. HuMI still
needs its own refactor before it can use the simplified trainer.
"""


if __name__ == "__main__":
    raise SystemExit("HuMI training is legacy right now. Use AaltoDB or HMOGDB, or refactor HuMI separately.")
