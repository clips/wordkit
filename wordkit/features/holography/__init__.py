"""Holographic features."""
from .kanerva import (KanervaLinearTransformer,
                      KanervaNGramTransformer,
                      KanervaOpenNGramTransformer)
from .plate import (PlateLinearTransformer,
                    PlateNGramTransformer,
                    PlateOpenNGramTransformer)


__all__ = ["KanervaLinearTransformer",
           "KanervaNGramTransformer",
           "KanervaOpenNGramTransformer",
           "PlateLinearTransformer",
           "PlateNGramTransformer",
           "PlateOpenNGramTransformer"]
