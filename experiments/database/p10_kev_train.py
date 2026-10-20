"""kev.train with Kev's own MPS shape bucketing applied during training as well (phase 10, 2026-10-17).

On Apple GPUs every new input shape gets its own compiled kernels, which stay in memory for the life of the process. kev.model already pads a
batch's rows to a multiple of DecisionModel.SHAPE_BUCKET (64 tokens; KEV_SHAPE_BUCKET) on MPS, but only in eval mode. In training every new row
length is a new shape, so a long run's memory grows with the number of distinct lengths it has seen (Kev-0.8B reached 40-44 GB over 8,000-10,000
records, Kev-4B passed 41 GB after 2,160). This launcher applies the same padding in training: the same pad token, position 0 and attention-mask 0
as Kev's eval path, after every real token, so a real token never attends to a pad (Kev: "parity measured exact") and no loss term reads one.
Everything else is kev.train unchanged; its arguments are passed through.

    .venv/bin/python <repo>/experiments/database/p10_kev_train.py kev.train --data ...   (as python -m kev.train --data ...)

The module is named on the command line, as with python -m, so the process still reads "kev.train" there: p10_run.sh's memory watchdog and its
pre-flight check find the training process by that name.
"""

import runpy
import sys

import torch

import kev.model as model

_pad_rows = model.DecisionModel._pad_rows


def pad_rows_bucketed(self, rows):
    ids, pos, att = _pad_rows(self, rows)
    if str(self.device) == "mps" and self.training:
        extra = -(-ids.shape[1] // self.SHAPE_BUCKET) * self.SHAPE_BUCKET - ids.shape[1]
        if extra:
            ids = torch.cat([ids, torch.full((ids.shape[0], extra), self.pad_id, dtype=ids.dtype, device=ids.device)], 1)
            pos = torch.cat([pos, torch.zeros((pos.shape[0], extra), dtype=pos.dtype, device=pos.device)], 1)
            att = torch.cat([att, torch.zeros((att.shape[0], extra), dtype=att.dtype, device=att.device)], 1)
    return ids, pos, att


model.DecisionModel._pad_rows = pad_rows_bucketed

if __name__ == "__main__":
    if sys.argv[1:2] != ["kev.train"]:
        sys.exit("usage: p10_kev_train.py kev.train <arguments of kev.train>")
    print(f"p10_kev_train: MPS shape bucketing in training, multiple of {model.DecisionModel.SHAPE_BUCKET} tokens", flush=True)
    sys.argv = sys.argv[1:]
    runpy.run_module("kev.train", run_name="__main__", alter_sys=True)
