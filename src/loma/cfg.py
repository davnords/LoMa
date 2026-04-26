from typing import Annotated
import tyro

from .loma import LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR, LoMa, LoMaBturbo

# Accept either a raw LoMa.Cfg instance or a named preset.
LoMaConfig = (
    Annotated[LoMaB128, tyro.conf.subcommand("loma_b128")]
    | Annotated[LoMaB, tyro.conf.subcommand("loma_b")]
    | Annotated[LoMaBturbo, tyro.conf.subcommand("loma_b_turbo")]
    | Annotated[LoMaL, tyro.conf.subcommand("loma_l")]
    | Annotated[LoMaG, tyro.conf.subcommand("loma_g")]
    | Annotated[LoMaR, tyro.conf.subcommand("loma_r")]
    | Annotated[LoMa.Cfg, tyro.conf.subcommand("loma_custom")]
)