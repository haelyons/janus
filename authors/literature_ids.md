# InceptionV1 neuron IDs from the literature (for ground truth)

Read 2026-10-21. Zoom In (distill.pub/2020/circuits/zoom-in/): IDs are svg element ids inside its figures
(id="mixed4b_237" etc.), grouped by the section heading above each figure; roles from the text and layout.
- Circuit 3, Cars in Superposition: car detector 4c:447 built from 4b:237 (top of the window: windows),
  4b:491 (middle: car body), 4b:373 (bottom: wheels) ("it looks for wheels at the bottom of its convolutional
  window, and windows at the top"). Readers shown: 4d:417, 4d:241, 4d:277.
- Circuit 2, Oriented Dog Head Detection: "two mirrored pathways, detecting analogous units facing to the left and
  to the right" over three layers. Left column / right column of the figure: 4a:234 / 4a:308, 4a:325 / 4a:222,
  4b:443 / 4b:431, 4b:429 / 4b:384, 4b:409 / 4b:418, 4c:254 / 4c:407. Inputs 3b:404, 3b:303. Which column faces
  which way is not in the text.
- Circuit 1, Curve Detectors: 3a:147, 81, 145, 71 into 3b:379, 385, 342, 330.
- Polysemantic example in the text: 4e:55 ("cat faces, fronts of cars, and cat legs").
High-Low Frequency Detectors (distill.pub/2020/circuits/frequency-edges/): fifteen in mixed3a; named 3a:136, 108,
132, 88, 110, 180, 153, 186, 86, 117, 112, 70, 106, 113. (The essay places them in 3b.)
Equivariance (distill.pub/2020/circuits/equivariance/): no IDs in text.
Essay neurons not studied before round 20: mixed5a:719 ("a pinched-center shape: bow ties, butterflies, swimsuit
tops", with semantic residue: dress shoes, beach bags, flip-flops); the 4b car parts; 447's readers ("two are car
neurons and two share a pose"); high-low frequency detectors; "more than a hundred" pose neurons.
