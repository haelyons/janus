You are advising on methodology for interpreting individual units (neurons) of trained neural networks. An observer (another AI) is shown recorded data from one unit and must say what it responds to; it cannot run experiments, it only reads what it is shown.

Suppose an automated profile shows that a unit of a convolutional image classifier is unusually sensitive to some transformation T of its input photos, compared with the other units of its layer (T could be a flip, a rotation, a colour change, a blur, a shuffle of image patches, and so on). Knowing that the unit is sensitive to T does not yet say which property of the photo matters: an observer can misattribute the sensitivity to the wrong property.

Design up to 5 generic follow-up tests or views, computed automatically in advance and usable for any T and any unit, that would let the observer tell which property of the photos T disrupts for this unit. For each: what to compute; what to show the observer; how to read the result; what it costs (labels, compute); and the analogue for a transformer language model, where T is a transformation of the input text. Rank by expected value.

Reply with only JSON: [{"rank": n, "name": "...", "compute": "...", "show": "...", "read": "...", "cost": "...", "text": "..."}]
