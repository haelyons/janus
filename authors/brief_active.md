Write one paragraph of instructions for a subject: a language model with no tools, in a single conversation.

The subject has just been shown recorded data from one unit of a language model: windows of text in which tokens are
marked «token|activation», activations scaled 0-10 relative to the strongest example shown. After your paragraph it
will be asked to say what the unit responds to, and later to predict the unit's activation on new text.

Your paragraph is inserted between the data and that final task. It offers the subject experiments: the subject may
write its own texts and see the unit's activation on every token, marked the same way. The paragraph must state these
rules exactly:
- at least 1 and at most 3 requests before the final answer; at most 5 texts per request;
- a request is a reply containing only one JSON object of this form:
  {"hypotheses": ["...", "..."], "show": [{"text": "...", "target": "the token being tested", "predictions": ["high" or "low", one per hypothesis]}]}
- results come back with every token's activation on the same 0-10 scale;
- when finished, the subject replies with the final answer in the format described after the paragraph, instead of
  a request.
Beyond these rules, write whatever instructions you think will make the subject's experiments most informative about
what the unit really responds to. At most 200 words. Reply with only the paragraph.
