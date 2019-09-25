"""Holographic features."""
from .kanerva import (KanervaLinearTransformer,
                      KanervaNGramTransformer,
                      KanervaOpenNGramTransformer,
                      KanervaConstrainedOpenNGramTransformer)
from .plate import (PlateLinearTransformer,
                    PlateNGramTransformer,
                    PlateOpenNGramTransformer,
                    PlateConstrainedOpenNGramTransformer)


__all__ = ["KanervaLinearTransformer",
           "KanervaNGramTransformer",
           "KanervaOpenNGramTransformer",
           "PlateLinearTransformer",
           "PlateNGramTransformer",
           "PlateOpenNGramTransformer",
           "KanervaConstrainedOpenNGramTransformer",
           "PlateConstrainedOpenNGramTransformer"]
