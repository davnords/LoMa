from typing import Annotated
import tyro

from .loma import (
    LoMaB,
    LoMaB128,
    LoMaL,
    LoMaG,
    LoMaR,
    AffineLoMaB,
    AffineLoMaL,
    AffineLoMaG,
    LoMa,
)

# Accept either a raw LoMa.Cfg instance or a named preset.
LoMaConfig = (
    Annotated[LoMaB128, tyro.conf.subcommand("loma_b128")]
    | Annotated[LoMaB, tyro.conf.subcommand("loma_b")]
    | Annotated[LoMaL, tyro.conf.subcommand("loma_l")]
    | Annotated[LoMaG, tyro.conf.subcommand("loma_g")]
    | Annotated[LoMaR, tyro.conf.subcommand("loma_r")]
    | Annotated[AffineLoMaB, tyro.conf.subcommand("affine_loma_b")]
    | Annotated[AffineLoMaL, tyro.conf.subcommand("affine_loma_l")]
    | Annotated[AffineLoMaG, tyro.conf.subcommand("affine_loma_g")]
    | Annotated[LoMa.Cfg, tyro.conf.subcommand("loma_custom")]
)
