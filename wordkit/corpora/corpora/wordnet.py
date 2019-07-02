"""Read semantic information from multilingual wordnets."""
from ..base import reader


def wordnet(path,
            language,
            restrict_pos=None,
            fields=("orthography",
                    "semantics")):
    """Get semantic information."""
    df = reader(path,
                fields=fields,
                field_ids={"semantics": 0,
                           "orthography": 2},
                language=language,
                comment="#",
                sep="\t",
                header=None)
    if restrict_pos:
        if not isinstance(restrict_pos, (tuple, list, set)):
            restrict_pos = {restrict_pos}
        df = df.where(semantics=lambda x: x.split("-")[1] in restrict_pos)
    return df
    return df.aggregate("orthography",
                        "semantics",
                        lambda x: x)
