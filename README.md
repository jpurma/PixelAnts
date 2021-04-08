# PixelAnts
Ant colony algorithm with Numpy and Kivy

A toy for testing if fast array handling from Numpy and easy access to OpenGL bitmaps from Kivy are enough for drawing hundreds of ants and their pheromon paths. Seems to be so.

Ant colony algorithm here is a bit unrealistic, but an interesting variation: foraging aka. outgoing ants leave one kind of scent signal and returning ants leave another kind of scent signal, and they always use the other kind of signal to navigate either towards food or towards home.

Outgoing signals are tuples of (travel time, direction), where if an ant encounters a signal where travel time (time since last visit to the hive) is longer than its current travel time, it leaves a new signal with its current direction replacing the old one. Outgoing signal are visualised with hue corresponding to direction. Because the signals get replaced with ones suggesting a faster route to that point, they have an effect of slowly painting the world with fast routes from hive to each point.

Returning or incoming signals are tuples of (decay, direction). These are drawn when ant has found food and is returning to hive by following outgoing signals to their reverse direction, or when it is traversing randomly when there is no outgoing signal present. There can be up to 10 returning signals on each point. Returning signals are used by outgoing/foraging ants, for each signal there is a chance for it to be chosen and followed into reverse direction. These signals decay over time, and when there are new signals, they pop out the older signals so that there are maximum ten signals. These are visualised with transparent white pixels.

There is also option for ants to use path integration to keep track on how far they are from hive and in which direction, and how far they are from their last known path point. This path integration uses cosine rule to calculate how current step has modified ant's position relative to previous step's calculation of hive position. It is known that at least some real ants use path integration to return directly to home, but in this simulation it often leads to traps, where ants get stuck in U-shapes and still draw routes, attracting more ants into same traps. Adjusting values and adding heuristics on when to attempt a direct path and when to rely on scent routes would improve it, but I'm more fascinated with pure scent routes.

Requires Python 3.8, Numpy and Kivy, then:

    python pixelants.py

