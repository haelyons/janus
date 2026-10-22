An image classification network trained on ImageNet has units (single neurons) that respond to features of photos.
To find out what a unit responds to, someone who never sees the unit's activity answers questions about ordinary
photos, and the answers are compared with the unit's activity among photos of the same class. Questions written from
a description of the unit tend to cover only what the description mentions.

We want a fixed set of extra questions, asked for every unit whatever its description: about ways in which photos of
the same class commonly differ that the network's training treats as irrelevant to the class, for example the
variations used in standard training-data augmentation. Write 8 such questions, about the photo's main subject or
the photo as a whole. Each must be answerable by looking at a small photo, and needs 2 to 5 short answers that do not
overlap, so that every photo fits exactly one of them; include an answer for photos to which the question does not
apply. Reply with only JSON: {"questions": [{"question": "...", "answers": ["...", "..."]}]}
