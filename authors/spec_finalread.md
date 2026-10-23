# Spec: tools/finalread.py (a final reading of a round-29 table that sees its picks' checks on new photos)

Write one new file, `tools/finalread.py`. Read only this spec, `tools/pairtable.py`, and in `tools/proploop.py` only
the functions it names: fill, _whole, attempt, cached_calls, save_json (find them with a search; do not read the
rest of proploop). `bundlemax.last_obj` and `contrastloop._text` are used as pairtable uses them. Do not open any
other file, and do not run anything.

## Inputs
- `runs/pairtable/<unit>/stats.json` for each unit of `--units-file` (default `authors/round29_units.json`), as
  written by pairtable.py: m, class_names, D, T, questions, tables["real"], readings, picks.
- Outputs in `runs/finalread/`.

## Calls (per unit, per s in 1, 2, 3 whose readings["real_s{s}"] is valid; arms "real" and "null")
Keys "real_s1" ... "real_s3", "null_s1" ... "null_s3". For key "{arm}_s{s}", the picks are picks["real_s{s}"] (the
rows picked from the real table, in order) and each pick's check is its "T" (arm real) or its "T_null" (arm null).
Content: one text block (contrastloop._text), these parts joined by a blank line ("\n\n"):

1. proploop.fill(pairtable.TABLE_HEAD, n=len(D), K=24, classes=", ".join(class_names), m=m) (whole numbers as
   pairtable.table_text).
2. The real table's rows, one pairtable.row_line(row, questions) per row, joined by "\n".
3. Verbatim, with {explanation} the reading's explanation and {rows} its picked row numbers joined by ", ":

   Someone who read this table wrote: "{explanation}" They picked rows {rows} as the ones expected to hold for this
   unit on new photos and to describe what is particular to it.

4. Verbatim with {n} = len(T), then, after a newline, one line per pick in order:

   The picked rows were then checked on {n} new photos of the same classes, none of them among those above,
   answered in the same way. A row held when this unit's d, with the same answer first, is at the 95th percentile of
   its layer or above.

   per pick: `Row {row}: "{first_answer}" ({n_first} photos) more than "{second_answer}" ({n_second} photos): d {d with
   a sign and 2 decimals}, percentile {pct}; similar units {p1}, {p2}, {p3}: {held | did not hold}.` (pct and peer
   percentiles whole by proploop._whole; "held" when the check's outcome is "held"), or, when the outcome is
   "too few": `Row {row}: "{first_answer}" ({n_first} photos) more than "{second_answer}" ({n_second} photos): too
   few to check (each needs at least 5).`

5. Verbatim:

   Taking all of this into account, say in a few sentences what this unit responds to, and list the rows of the
   table, by number, that you are confident describe this unit. Reply with only JSON: {"explanation": "...",
   "rows": [n, ...]}

Parse: as pairtable.parse_rows(text, number of rows of the real table), except that the list may be empty and no
cap applies: valid when "explanation" is a non-empty string after strip and "rows" is a list; keep its ints (not
bool) from 1 to the number of rows, in order, first occurrence only. One retry (proploop.attempt) when invalid.
Through proploop.cached_calls, one call at a time (main sets proploop.WORKERS = 1 before any call), cache
`runs/finalread/<unit>/readings.json`.

## Outputs
- `runs/finalread/<unit>/stats.json`: unit, and per key {"valid", "explanation", "rows", "picked" (the picked row
  numbers), "held" (picked rows whose check for that arm held)}; valid false and the others None when not valid or
  not made.
- `runs/finalread/report.txt`, printed: per unit and key, picked rows with their outcome for that arm, final rows,
  and the explanation (first 300 characters).

## Interface (a separate tester writes tests against these)
- `content_text(st, s, arm)`: st the stats.json dict; the text of the content above.
- `parse_final(text, n_rows)`: {"explanation", "rows"} by the rule above, or None.
- `check_line(pick, chk)`: one line of part 4 (chk the pick's "T" or "T_null" dict).

## Selftest
`--selftest`, no Claude: check_line for held, did not hold and too few; parse_final (empty list valid, bools,
duplicates, out of range, missing explanation); content_text on a small hand-made st (part order, the arm's checks
used). Print `selftest ok`.

## Usage line
    python -I tools/finalread.py [--units-file authors/round29_units.json] [--selftest]
