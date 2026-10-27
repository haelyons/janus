Write test sentences for a language-model experiment. Reply with only one JSON object, no prose.

We need 32 "forms": ways of saying something that vary in surface form but not in what event is reported. Each form is
written in four parallel versions that differ ONLY in the event:
- "postponed": the meeting was postponed (put off to a later time)
- "cancelled": the meeting was cancelled
- "moved_earlier": the meeting was moved to an earlier time
- "flight_delayed": the flight was delayed
Within a form, the four versions must be identical apart from the words that describe the event (and the subject noun
for flight_delayed), so that form and event can be separated.

Across forms, vary the surface as much as you can while keeping the reported event: hedges and epistemic markers in
different positions ("(I think)", ", AFAIK", " AFAIK", " IIRC", "I think ...", "apparently", "probably"); yes/no and
wh-questions about the event ("was the meeting postponed?", "why was the meeting postponed?"); active and passive
voice ("they postponed the meeting"); different words for the same event ("pushed back", "put off", "delayed",
"called off", "scrapped", "brought forward"); tense and aspect ("has been", "had been", "got"); reported speech
("she said the meeting was postponed"); short and longer sentences; informal and formal register; punctuation and
casing. Each sentence at most 20 words, English, no line breaks.

Format:
{"forms": [{"id": 1, "stance": "plain" | "hedged", "speech_act": "assertion" | "yes_no_question" | "wh_question" |
"reported", "voice": "passive" | "active" | "other", "versions": {"postponed": "...", "cancelled": "...",
"moved_earlier": "...", "flight_delayed": "..."}}, ...]}
