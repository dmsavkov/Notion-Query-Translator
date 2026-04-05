#!/usr/bin/env python
"""Flash sandbox Notion databases without reseeding baseline records."""

from src.evaluation.sandbox import provision_infrastructure


if __name__ == "__main__":
    provision_infrastructure()
