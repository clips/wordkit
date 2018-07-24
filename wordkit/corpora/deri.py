"""Tools for working with the multilingual corpora from Deri and Knight."""
import re

from .base import Reader, segment_phonology, diacritics
from ipapy.ipastring import IPAString


ALLOWED_LANGUAGES = {'fas',
                     'isl',
                     'kld',
                     'tzm',
                     'afr',
                     'ind',
                     'epo',
                     'rue',
                     'ile',
                     'ruq',
                     'umb',
                     'uzb',
                     'hts',
                     'wlm',
                     'yua',
                     'bzq',
                     'tel',
                     'sjd',
                     'unm',
                     'bos',
                     'ain',
                     'ccc',
                     'smn',
                     'ukr',
                     'ady',
                     'bcl',
                     'cym',
                     'nno',
                     'eus',
                     'yox',
                     'msa',
                     'bel',
                     'ces',
                     'gsw',
                     'mkd',
                     'lao',
                     'bre',
                     'mal',
                     'abk',
                     'pol',
                     'heb',
                     'arc',
                     'yue',
                     'guj',
                     'sco',
                     'xho',
                     'kat',
                     'pan',
                     'kor',
                     'ltz',
                     'bdr',
                     'glv',
                     'nds',
                     'sei',
                     'nan',
                     'got',
                     'slk',
                     'zlm',
                     'fij',
                     'aht',
                     'btf',
                     'tur',
                     'zha',
                     'pon',
                     'kab',
                     'cjs',
                     'kaa',
                     'urb',
                     'agm',
                     'brg',
                     'hil',
                     'ett',
                     'chu',
                     'mya',
                     'arg',
                     'ayl',
                     'bbc',
                     'nor',
                     'run',
                     'vie',
                     'ceb',
                     'hif',
                     'twf',
                     'alp',
                     'akz',
                     'hsb',
                     'san',
                     'yai',
                     'kac',
                     'pal',
                     'aby',
                     'cat',
                     'sah',
                     'apw',
                     'ale',
                     'dan',
                     'osx',
                     'som',
                     'gle',
                     'udm',
                     'aia',
                     'sme',
                     'tuk',
                     'wln',
                     'tts',
                     'syc',
                     'hye',
                     'hdn',
                     'byq',
                     'cmn',
                     'gvf',
                     'lit',
                     'tgk',
                     'bar',
                     'bnn',
                     'akr',
                     'ank',
                     'dob',
                     'kay',
                     'lav',
                     'gml',
                     'kin',
                     'zpq',
                     'ina',
                     'swa',
                     'kaz',
                     'pus',
                     'lij',
                     'myv',
                     'gla',
                     'hin',
                     'mon',
                     'srp',
                     'stp',
                     'nrf',
                     'ell',
                     'ast',
                     'nav',
                     'mlt',
                     'tat',
                     'unk',
                     'nob',
                     'swe',
                     'enm',
                     'frm',
                     'ava',
                     'glg',
                     'bal',
                     'iii',
                     'itl',
                     'dlm',
                     'luo',
                     'sdh',
                     'sin',
                     'tvl',
                     'tso',
                     'sga',
                     'yid',
                     'ron',
                     'jbo',
                     'kyu',
                     'sat',
                     'wol',
                     'goh',
                     'nah',
                     'nhx',
                     'bul',
                     'ben',
                     'ltc',
                     'mdf',
                     'dru',
                     'arz',
                     'deu',
                     'scn',
                     'tpi',
                     'dum',
                     'vol',
                     'mul',
                     'fry',
                     'kgp',
                     'spa',
                     'tha',
                     'xcl',
                     'que',
                     'khm',
                     'tlh',
                     'chv',
                     'uig',
                     'tyv',
                     'sot',
                     'squ',
                     'hbs',
                     'grc',
                     'sqi',
                     'ewe',
                     'ang',
                     'wuu',
                     'aze',
                     'xal',
                     'mtq',
                     'xfa',
                     'bjn',
                     'cri',
                     'frp',
                     'krc',
                     'oss',
                     'zul',
                     'alt',
                     'fin',
                     'ach',
                     'cha',
                     'chy',
                     'tfn',
                     'kbd',
                     'aar',
                     'haw',
                     'mhn',
                     'xbc',
                     'naq',
                     'cia',
                     'chm',
                     'est',
                     'zza',
                     'yrk',
                     'chk',
                     'pua',
                     'arn',
                     'nld',
                     'tgl',
                     'kur',
                     'cor',
                     'mlg',
                     'kij',
                     'tsi',
                     'lat',
                     'aka',
                     'ido',
                     'mng',
                     'mri',
                     'cos',
                     'hun',
                     'mww',
                     'osp',
                     'aek',
                     'ban',
                     'dsb',
                     'lad',
                     'ltg',
                     'str',
                     'khb',
                     'tsn',
                     'tli',
                     'fra',
                     'kan',
                     'nap',
                     'roh',
                     'zho',
                     'egy',
                     'bak',
                     'lnd',
                     'tam',
                     'amm',
                     'rom',
                     'vep',
                     'amh',
                     'csb',
                     'ita',
                     'bod',
                     'hrv',
                     'isd',
                     'nci',
                     'jpn',
                     'por',
                     'oci',
                     'crh',
                     'rap',
                     'unknown',
                     'aud',
                     'cic',
                     'eng',
                     'lez',
                     'ofs',
                     'bua',
                     'urd',
                     'rus',
                     'lim',
                     'bis',
                     'chr',
                     'sms',
                     'ami',
                     'egl',
                     'ryu',
                     'ppl',
                     'gag',
                     'hrx',
                     'slv',
                     'fao',
                     'tpw',
                     'ara'}

language2field = {'orthography': 2, 'phonology': 3, 'language': 0}


class Deri(Reader):
    r"""
    The reader for the Deri and Knight series of corpuses.

    These corpora are described in the paper:
    Grapheme-to-Phoneme Models for (Almost) Any Language
    by Deri and Knight (2016).

    This reader is different from the others because all the languages
    are in a single file. We still allow the user to only specify a single
    language per reader.

    The set of corpora can be downloaded here:
        https://drive.google.com/drive/folders/0B7R_gATfZJ2aSlJabDMweU14TzA

    If you use this corpus reader or the corpora, you _must_ cite the following
    paper:

    @inproceedings{deri2016grapheme,
      title={Grapheme-to-Phoneme Models for (Almost) Any Language.},
      author={Deri, Aliya and Knight, Kevin}
    }

    Parameters
    ----------
    path : string
        The path to the corpus this reader has to read.

    language : string, default ("eng")
        The language of the corpus. If this is set to None, all languages
        will be retrieved from the corpus.

    fields : iterable, default ("orthography", "phonology")
        An iterable of strings containing the fields this reader has
        to read from the corpus.

    merge_duplicates : bool, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        If this is False, duplicates may occur in the output.

    diacritics : tuple
        The diacritic markers from the IPA alphabet to keep. All diacritics
        which are IPA valid can be correctly parsed by wordkit, but it may
        not be desirable to actually have them in the dataset.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "phonology"),
                 language=None,
                 merge_duplicates=True,
                 diacritics=diacritics):
        """Extract words from Deri and Knight corpora."""
        super().__init__(path,
                         fields,
                         language2field,
                         language,
                         merge_duplicates,
                         diacritics=diacritics)

        if language is not None and language not in ALLOWED_LANGUAGES:
            raise ValueError("The language you supplied is not in the list "
                             "of allowed languages for this corpus.")
        self.matcher = re.compile(r"([:/]|rhymes)")

    def _retrieve(self, iterable, wordlist=None, **kwargs):
        """
        Extract sequences of phonemes for each word from the databases.

        Parameters
        ----------
        wordlist : list of strings or None.
            The list of words to be extracted from the corpus.
            If this is None, all words are extracted.

        Returns
        -------
        words : list of dictionaries
            Each entry in the dictionary represents the structured information
            associated with each word. This list need not be the length of the
            input list, as words can be expressed in multiple ways.

        """
        use_p = 'phonology' in self.fields

        wordlist = set([x.lower() for x in wordlist])
        words_added = set()

        for line in iterable:

            line = line.strip()
            columns = line.split("\t")
            if self.language is not None and columns[0] not in self.language:
                continue
            orthography = columns[self.field_ids['orthography']].lower()

            word = {}

            if wordlist and orthography not in wordlist:
                continue
            m = self.matcher.finditer(orthography)
            try:
                next(m)
                continue
            except StopIteration:
                pass
            words_added.add(orthography)
            word['orthography'] = "_".join(orthography.split())
            if use_p:
                syll = columns[self.field_ids['phonology']].split()
                syll = "".join(syll)
                try:
                    syll = "".join([str(x)
                                    for x in IPAString(unicode_string=syll)])
                except ValueError:
                    pass

                word['phonology'] = segment_phonology(syll,
                                                      to_keep=self.diacritics)
            if 'language' in self.fields:
                word['language'] = columns[self.field_ids['language']]

            yield word
