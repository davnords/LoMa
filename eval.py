from typing import Literal
import os

import tyro

from loma import LoMa, LoMaB
from loma.loma import LoMaConfig
from loma.random import set_seed
from loma.benchmarks import (
    Mega1500,
    ScanNet1500,
    RubikBenchmark,
    WxBSBenchmark,
)


def main(
    matcher: LoMaConfig = LoMaB(),
    benchmark: Literal[
        "mega1500",
        "scannet1500",
        "rubik",
        "wxbs",
    ] = "mega1500",
):
    set_seed(1337)
    model = LoMa(matcher)
    if benchmark == "mega1500":
        mega1500 = Mega1500()
        res = mega1500.benchmark(model)
        print(res)
    elif benchmark == "scannet1500":
        scannet1500 = ScanNet1500()
        res = scannet1500.benchmark(model)
        print(res)
    elif benchmark == "rubik":
        rubik = RubikBenchmark()
        res = rubik.benchmark(model)
        print(res)
    elif benchmark == "wxbs":
        wxbs = WxBSBenchmark()
        res = wxbs.benchmark(model)
        print(res)
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    matcher_name = getattr(matcher, "name", type(matcher).__name__.lower())
    os.makedirs("results", exist_ok=True)
    with open(f"results/{matcher_name}_{benchmark}.json", "w") as f:
        import json

        json.dump(res, f, indent=4)


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.CascadeSubcommandArgs,))
