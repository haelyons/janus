I want your advice on a research question. Please stay high-level.

## The goal

Nick Cammarata's recent essay "Neural Networks Are Already Clean" (Goodfire, October 2026) argues that the field misread neural networks. For years, neurons that seemed to respond to unrelated things were taken as evidence of superposition: networks packing unrelated concepts into one neuron, so that neurons were the wrong unit to study. He now argues that most such neurons have one job, sometimes with related offshoots, and that the apparent mixtures came from how we looked, not from the networks. Some of his examples:

- The famous "cat and car" neuron in an image model. For years it was described as responding to cat faces OR car fronts. He looked at the image grid he had seen hundreds of times, spent a few seconds asking "what one thing might this be?", and saw a V-shape detector. Its strongest inputs include V-shaped flames, fireboat water jets and fox faces. Its negative side is round things, an O against the V. Dogs and cars were also grouped elsewhere in the network by shared pose, and that had been read as "a car hidden in dog neurons".
- A language-model neuron described in a well-known paper as firing on "academic citations, English dialogue, HTTP requests, and Korean text". He reads it as a "don't say a dot" neuron: nearly all its firings are on or near a dot, and most of its strongest negative output connections are tokens beginning with a dot.
- A neuron that fired on "pr" and also on Base64 URL strings and Bengali text. Decoded or transliterated, those were "pr" too.
- A "polysemantic comma neuron" that tracked Colorado. It stored a running summary of earlier context on punctuation, and that context was out of view in short excerpts.
- Neurons whose "unrelated jobs" were really predicting the next token (a Titanic neuron detecting that "boat" comes next), or whose meaning sat on the negative side (Santa lore), or noise from a sign-regime quirk in the nonlinearity.
- His own account: for years he treated superposition as the null hypothesis when it should have been the last. It "just looks like things that don't make sense", and his belief in it kept explaining away the evidence that would have overturned it. "If you assume a network is clean and you're wrong, a messy neuron is a lead... If you assume it's packed and you're wrong, a messy neuron is the end."

We want to know whether this kind of shift can be produced reliably in a capable model by prompting: the same data read differently, the way a fresh mind sees what an expert looked past. We care about the shift itself, in a way that could transfer to other fields, not about neuron interpretation as such. We want as little imposed method as possible. A checklist of Nick's tricks is not the goal. Success has to be the correct reading, not just the opposite prior: a model told "neurons are clean" that then finds unity everywhere has learned nothing.

## What we have observed so far

All of these use a current frontier chat model. Each case is a single unit with recorded data, and we score its predictions on new inputs it has not seen.

1. **The cat/car neuron itself, recorded fresh on a different photo set.** Its 20 strongest photos are mostly pointed-ear faces (foxes, cats, red pandas, lynx, a hamster, a sheltie) and one red convertible. Its 20 most negative photos are mostly round, flat or jowly dog faces (pugs, mastiffs, a boxer), a golf ball and a soccer ball. On new photos it also fires strongly on jeeps, cabs, pool tables and a throne.
   - Shown the top photos, the model says "faces of small furry mammals", sometimes adding "plus an unrelated car".
   - Shown the negatives as well, a few answers note that flat-faced dogs push it down. All still say "small furry mammal faces".
   - Nobody predicts that cars, pool tables or the throne will fire.
2. **Varying the instructions changed almost nothing.** We tried no instruction at all; a 2023-style "summarise what the neuron is looking for"; the superposition consensus stated as background; Nick's thesis stated as background; a carefully written "look as if for the first time, let the data decide" prompt; and a prompt written from Nick's essay describing his methods. All gave nearly the same readings and the same prediction accuracy on this neuron.
3. **Then we let the model ask for more data before answering:** name image categories and see the neuron's activation on new photos of them, up to three times.
   - Every session used it. They asked for convertibles, sports cars, cabs, pickups, more cats, foxes and some dogs.
   - Every session concluded "cat- and fox-like faces, and separately cars: unrelated". One said "no visual feature obviously shared". That is the field's original reading, rediscovered.
   - Predictions improved only on cars.
   - No session asked for anything that is neither an animal nor a vehicle. None asked what cars and those faces might have in common.
4. **We also built text units where the truth is known exactly.**
   - Models correctly read units defined by a structural position: "the token before a dot that doesn't end a sentence"; "the last token of a list item".
   - They read a word with a related offshoot correctly: coffee, with weaker cafés and "Java".
   - They correctly called genuinely unrelated OR-combinations unrelated (cheese names or C preprocessor directives).
   - On a unit whose deciding context was outside the visible window, no model ever said "cannot tell". All gave a confident wrong reading, with predictions near chance.
   - Offered the same "ask for more" option on text, they never used it.
5. **From separate work:** chat-trained models are almost incapable of saying "I don't know". This held under fake facts and under being gaslit.

## The question

How would you elicit this kind of paradigm shift: seeing one thing where the received reading sees two, and doing so only when it is true? What are we missing? Keep it high-level: what is actually happening in the leap, what would trigger it, and what would show we had triggered it rather than imposed it.

Don't answer until you've thought about the problem as thoroughly as possible without answering
