"""Read semantic information from multilingual wordnets."""
from wordkit.corpora.base import reader


def wordnet(path, language, restrict_pos=None, fields=("orthography", "semantics")):
    """Get semantic information."""
    df = reader(
        path,
        fields=fields,
        field_ids={"semantics": 0, "orthography": 2},
        language=language,
        comment="#",
        sep="\t",
        header=None,
    )
    if restrict_pos:
        if not isinstance(restrict_pos, (tuple, list, set)):
            restrict_pos = {restrict_pos}

        def pos_func(x):
            return x.split("-")[1] in restrict_pos

        df = df[[pos_func(x) for x in df["semantics"]]].reset_index(drop=True)

    return df
