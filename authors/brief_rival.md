Write a system prompt for an AI agent.

The agent will be shown recorded data from a single unit (one neuron) of a trained neural network: either windows of text with the unit's activation marked on tokens, or a grid of images with the unit's activation on each. It must say what the unit does. The user message will specify a fixed JSON reply: a class (one thing / one main thing plus related extensions / two or more unrelated things / the data is not enough to tell), a short explanation, the evidence, what data is missing, and a confidence. In later turns it is shown new inputs and must predict, for each, whether the unit fires strongly or barely at all.

For image units, before answering, the agent may run a few rounds of experiments: it requests edits of photos in the grid (for example occluding a region, keeping only a region, zooming, mirroring, rotating, removing colour, and similar transformations) and sees the unit's activation on each edited photo. For text units it has no experiments and reads the data only. Its answers will be checked against exact ground truth about what the unit does, including its predictions on the new inputs.

Goal: the agent should work like a careful experimental scientist, so that its explanation is the one best supported against the alternatives, not merely one consistent with the data it happened to see first.

Return only the system prompt text, under 400 words.
