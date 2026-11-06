"""Native training with V2 defaults; no approval bypass."""
from experiments.qa.entry import train

if __name__ == "__main__":
    raise SystemExit(train("a2_2"))
