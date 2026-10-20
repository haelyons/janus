You are designing a characterization battery for units (channels) of a convolutional image classifier trained on ImageNet. The battery will be run identically on every channel of one mid-level layer (several hundred channels), each measured on its own most strongly activating natural photos. For each channel and each transformation, we record how much of the channel's activation survives when the photo is transformed.

List 6 to 8 image transformations for this battery. Requirements:
- Each is applied to a whole 224x224 RGB photo and returns a 224x224 RGB image.
- Each is deterministic (fix any random seed) and implementable in a few lines with PIL, numpy or scipy only (no other trained networks).
- Each is standard in vision science or in the analysis of CNNs, so the choice is defensible to a reviewer.
- Together they should be informative about a wide range of possible unit selectivities, without favouring any particular kind.

Return only JSON: {"transforms": [{"name", "how" (exact operation and parameters, precise enough to implement without questions), "removes", "keeps", "source" (a paper or standard practice)}]}.
