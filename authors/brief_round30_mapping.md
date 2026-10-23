# Brief: what does each description predict about photos?

Below are short descriptions of what individual units (single neurons) of an image classification network respond
to, written by researchers. Each was written about one unit's strongest responses. Below them are 8 questions that
someone answers about a photo, each with fixed answers.

For each description, say which of the 8 questions it predicts the unit's activity to depend on, among photos of the
kinds of things the unit responds to, and how: which answers should go with higher activity ("more") and which with
lower ("less"). Only include a question when the description itself says or directly implies it (for example a
description that names a viewpoint, a position in the frame, cropping, size in the frame, lighting, colour cast,
tilt or blur). Do not guess beyond the description. When a description says the unit prefers one of two mirror-image
cases but not which, give the two answers in "more" and "less" in either order and set "either" to true. Many
descriptions will predict no question; give an empty list for those.

Also give 2 to 6 keywords: single words, lower case, that a description of the unit agreeing with this one would very
likely contain (for example the main part, shape or object named).

Questions:
Q1. Which way is the main subject facing or pointing? Answers: "left"; "right"; "toward or away from the camera"; "no clear facing direction or no single main subject"
Q2. How much of the photo does the main subject take up? Answers: "less than a quarter"; "a quarter to three quarters"; "more than three quarters"; "no single main subject"
Q3. Is the main subject cut off by an edge of the photo? Answers: "fully inside the photo"; "partly cut off"; "no single main subject"
Q4. Where is the center of the main subject left to right? Answers: "left third"; "middle third"; "right third"; "no single main subject"
Q5. How brightly lit is the main subject? Answers: "dark"; "medium"; "bright"; "no single main subject"
Q6. Does the photo have an overall color tint? Answers: "no noticeable tint"; "warm (yellow, orange or red) tint"; "cool (blue or green) tint"; "black and white or nearly colorless"
Q7. Are the horizon or the main straight lines in the photo tilted? Answers: "level"; "noticeably tilted"; "no horizon or straight lines visible"
Q8. How sharp is the main subject? Answers: "sharp"; "slightly blurry"; "very blurry"; "no single main subject"

Descriptions:
L1. lower side/body of road vehicles, especially red fire [the label is cut off here]
L2. dog-like frontal muzzle / vehicle-front face pattern
L3. respirator-like front: paired round upper parts over [the label is cut off here]
L4. boxy orange/yellow vehicle-cab or rigid wheeled frame
L5. high-contrast striped/checkered lateral vehicle-or-wi [the label is cut off here]
L6. horizontal row of dark apertures/windows on a bright [the label is cut off here]
L7. boxy road-vehicle side with horizontal band/window st [the label is cut off here]
L8. Detects dog heads in one orientation. It belongs to two mirrored pathways of units, one detecting dog heads facing to the left and the other dog heads facing to the right; which of the two this unit belongs to is not stated. Detects dog snouts or heads in one orientation; another unit is a horizontally flipped version of it. Which way this unit faces is not stated.
L9. rounded painted vehicle/container body, train-like ma [the label is cut off here]
L10. small ground-contact circular/foot blobs: wheels and [the label is cut off here]
L11. side-profile vehicle lower body / wheelbase band
L12. side-view car/van body with horizontal window/beltline
L13. large wheeled utility vehicle lower-side / wheelbase
L14. hard engineered apparatus parts: vertical masts/panel [the label is cut off here]
L15. dark-centered circular hub / wheel-like radial object
L16. Detects dog heads in one orientation. It belongs to two mirrored pathways of units, one detecting dog heads facing to the left and the other dog heads facing to the right; which of the two this unit belongs to is not stated.
L17. Detects dog heads. It belongs to a set of dog-head units that come in variants of the same feature, such as short-snouted versus long-snouted dogs, or a human versus a dog version; which variant this unit is is not stated.
L18. warm high-contrast rectangular printed labels / signs
L19. wheel / circular badge-like dark disk on an object si [the label is cut off here]
L20. vehicle cab / side-window-on-colored-body detector
L21. side-on marked vehicle flank / police-car-livery dete [the label is cut off here]
L22. boxy bus/tram or large-van side-front: window band ov [the label is cut off here]
L23. vehicle wheel-arch / dark rounded inset in a hard bod [the label is cut off here]
L24. pale side-on body with dark windows/stripe/base, espe [the label is cut off here]
L25. dark rounded nose-or-wheel embedded in a rounded fron [the label is cut off here]
L26. compact side-view car / small quadruped body-face ali [the label is cut off here]
L27. mustelid-like furry body with pale masked snout / dar [the label is cut off here]
L28. framed rectangular display/window/readout panel
L29. dark head/lens/cab end on a pale or yellow object
L30. front-facing round porthole / circular appliance door [the label is cut off here]
L31. boxy vehicle window/cab band
L32. right-edge cropped vehicle / rounded object continuat [the label is cut off here]
L33. bright painted ground-vehicle side / school-bus flank
L34. compact side-view utility vehicle/boat body on open b [the label is cut off here]
L35. side-on bus/tram body with a horizontal row of rectan [the label is cut off here]
L36. front-facing rounded vehicle/animal face: central nos [the label is cut off here]
L37. boxy road-vehicle side/front body (bus/fire-truck/amb [the label is cut off here]
L38. compact rounded animal/vehicle/object front with dark [the label is cut off here]
L39. ornate rounded front/container with side lobes
L40. face-like dark-spot arrangement on animal heads or ve [the label is cut off here]
L41. side-view wheeled / spoked thin-frame assembly
L42. dark wheel / circular hub under an arch
L43. side-on vehicle lower body / undercarriage band
L44. wheel-like circular disk / annular hub detector
L45. dark rounded disk/blob in a bright curved surround
L46. large wheeled vehicle chassis / tractor-locomotive bo [the label is cut off here]
L47. dark circular wheel/tire annulus
L48. boxy vehicle side / bus-van body with horizontal wind [the label is cut off here]
L49. round disk / bullseye-like circular object detector
L50. wheel-like annular hub / circular port detector
L51. flat dark dog muzzle / brachycephalic dog face
L52. ambulance / emergency-service vehicle side with red-o [the label is cut off here]
L53. light box-truck / trailer side panel detector
L54. Detects a human beside an animal. Another unit is its horizontally flipped version: one responds to the human on one side and the animal on the other, its counterpart to the reverse arrangement. Which arrangement this unit prefers is not stated.
L55. man-made roofline / awning-top edge under sky
L56. smooth horizontal vehicle-body / bumper-side panel
L57. road vehicle body/front-corner (cars, trucks, vans)
L58. car / van front-cab and hood with windshield
L59. road-vehicle body in side/front-quarter view
L60. side-view wheel / vehicle undercarriage with round hub
L61. arched wheel-well / dark semicircular opening under a [the label is cut off here]
L62. round lens/headlight/wheel-like disk on a boxy mechan [the label is cut off here]
L63. boxy emergency/commercial vehicle side-front with whe [the label is cut off here]
L64. side-view land vehicle body / chassis
L65. side-on segmented / repeated-upright object patch, of [the label is cut off here]
L66. dark wheel or circular aperture under an arched/boxy [the label is cut off here]
L67. radial circular hub / wheel-or-flower center
L68. boxy road-vehicle side / window-belt detector
L69. exposed side-view vehicle wheel / wheel-well under ch [the label is cut off here]
L70. side-view wheel / bullseye-like circular hub-and-rim [the label is cut off here]
L71. dark/metal rounded hardware cap: microphone/thimble/w [the label is cut off here]
L72. rounded cup/pot/container rim, often terracotta/red
L73. side-view vehicle body / window-band detector
L74. front wheel / lower fender of boxy road vehicles
L75. rounded vehicle wheel/fender or vertical cylinder pat [the label is cut off here]
L76. side-view vehicle wheel / tire-and-fender patch
L77. concentric circular wheel/lens/hub annulus
L78. framed pale glass/window compartment, aliasing hourgl [the label is cut off here]
L79. clustered rounded tubes / spokes (pipe, wheel-hub, in [the label is cut off here]
L80. side-view bicycle / motorcycle wheel-and-frame assemb [the label is cut off here]
L81. boxy vehicle lower side / undercarriage band
L82. side-on bus/vehicle window band with horizontal dark [the label is cut off here]
L83. large vehicle lower-side / bus-body strip
L84. boxy light-colored van/ambulance/RV side-front with d [the label is cut off here]
L85. dark annular wheel / circular hub-aperture
L86. side-view vehicle lower body / hull over dark underca [the label is cut off here]
L87. warm-colored vehicle cab/window-and-hood corner
L88. isolated nested square/rectangular insert or display [the label is cut off here]
L89. horizontal axle with dark circular wheels/plates
L90. side-view vehicle wheel / radial tire-and-hub
L91. wheel/dial-like round parts on rigid side-on objects, [the label is cut off here]
L92. side-on rectangular road-vehicle body / bus-van-truck [the label is cut off here]
L93. small fluffy dog face / face-like eye-nose-muzzle pat [the label is cut off here]

Do not read any file or use any tool. Reply with only JSON, one entry per description, in this form:
{"L1": {"relations": [{"question": 1, "more": ["..."], "less": ["..."], "either": false}], "keywords": ["...", "..."]}, ...}
Use the answer texts exactly as written above.
