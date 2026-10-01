# Data files

## Columns

- `id`: item identifier (see below).
- `label`: `0` negative, `1` neutral, `2` positive (`3` mixed in `narabizi_4class`).
- `label_name`: the same label as text.
- `source_label`: the label in the original release (five levels `0`-`4`, very negative to very positive, for YouTube).
- `video`: YouTube video id of the comment (`novideo-<n>` when the release gives no URL).
- `text` (in `processed/` only): minimally cleaned text. HTML entities decoded; URLs, user mentions and the retweet marker
  removed; hashtags kept without `#`, with underscores turned into spaces; emoji, Arabizi digits, elongation and French
  words kept. The raw text is not redistributed.

## Identifiers

- `twifil-<n>`: row `n` (0-based) of `data/train_sent.csv` followed by `data/test_sent.csv` in the DziriBERT repository
  (github.com/alger-ia/dziribert, commit `8d6959ccec1cb9c475304ea97d0b68dff55c4377`): rows 0-7,076 are the training tweets and 7,077-9,436 the
  test tweets. The development set is a stratified 10% of the training tweets (seed 42).
- `yt-<n>`: row `n` (0-based) of the 45,000-comment CSV of the Mendeley release (V2) after dropping rows without text or
  label (none were dropped in V2).
- `narabizi-<split>-<sent_id>`: `sent_id` of the sentence in the NArabizi treebank, with its original split.

## Audit

- `audit/cleaning_audit.csv`: number of items removed by each audit step, per split.
- `audit/dataset_stats.csv`: size, class balance, script, length and emoji statistics of every version.
- `audit/removed_items.csv`: every removed item with its original split and the step that removed it:
  `1_empty_after_url_mention_removal` (no Arabic or Latin letter, emoji or emoticon left after cleaning; for TWIFIL the
  `detail` column says whether the tweet held only links and mentions, only punctuation or digits, or text in another
  script), `2_conflicting_labels` (all copies of a normalised text that occurs with more than one label),
  `3_duplicate_within_split` (repeats within a split, the first copy is kept) and `4_cross_split_leakage` (texts present
  in more than one split; the test copy, then the development copy, is kept).
  The TWIFIL rows refer to the cleaned version; the DziriBERT split itself is not audited. The 313 mixed NArabizi
  sentences dropped from the three-class version are a label filter, not an audit step, and are not listed.
