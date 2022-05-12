# New packages need to download

can manually download 'averaged_perceptron_tagger'&'universal_tagset'

# How to use the template

Each line represents a rule for a word that needs to be replaced

1. For column "key": Fill in the word we want, e.g. uob.
2. For column "pos_before": The lexical form of a word preceding that word, e.g. ADP.
3. For column "pos_after": The lexical form of a word that follows that word, e.g. VERB.
4. For column "replace": STT may be identified as, e.g. your bb;you will be.
5. For column "flag_direct_replace": If you want to directly replace some words without considering the part of speech, mark "x" here.

Note: In the 'pos_before', 'pos_after' and 'replace' columns, if there is more than one possibility, separate each with ';' and do not add spaces before or after the ';'.

e.g. your bb;your obey;you will be

e.g. ADP;VERB
