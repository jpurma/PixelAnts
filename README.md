# PixelAnts
Ant colony algorithm with [Numpy](https://numpy.org) and [Kivy](https://kivy.org).

This is a toy simulation for testing if the fast array handling from Numpy and the easy access to OpenGL bitmaps from Kivy are enough when drawing hundreds of ants with their pheromone paths. Seems to be so.

<img width="477" alt="screenshot" src="https://user-images.githubusercontent.com/5269272/114096061-e520a300-98c6-11eb-8307-4a6864321c01.png">

Ant colony algorithm here is a bit unrealistic, but interesting variation: foraging aka. outgoing ants leave one kind of a scent signal and returning ants leave another kind of a scent signal, and they always use the other kind of signal to help themselves navigate either towards food or towards home.

Outgoing signals are tuples of (travel time, direction), where if an ant encounters a signal where travel time (time since last visit to the hive) is longer than its current travel time, it leaves a new signal with its current direction replacing the old one. Outgoing signal are visualised with hue corresponding to direction. Because the signals get replaced with ones suggesting a faster route to that point, they have an effect of slowly painting the world with fast routes from hive to each point.

Returning or incoming signals are tuples of (decay, direction). These are drawn when an ant has found food and is returning to the hive by following outgoing signals to their reverse direction, or when there is no outgoing signal present and it is traversing randomly. There can be up to 10 returning signals on each point. Returning signals are used by outgoing/foraging ants, for each signal there is a chance for it to be chosen and followed into reverse direction and towards food. These signals decay over time, and when there are new signals, they pop out the older signals so that there are maximum ten signals at each point. These are visualised with transparent white pixels.

There is also an option for ants to use path integration to keep track on how far they are from the hive and in which direction, and how far they are from their last known path point. This path integration uses the cosine rule to calculate how the current step has modified an ant's position relative to the previous step's calculation of hive position. It is known that at least some real ant species use path integration to return directly to hive, but in this simulation it often leads to 'tar pits', where ants get stuck in U-shapes and keep drawing routes, attracting more ants into the same trap. Adjusting values and adding heuristics for when to attempt a direct path and when to give up and rely on scent routes would improve it, but I'm currently more fascinated with pure scent routes.

Requires Python 3.8, Numpy and Kivy,

    pip install -r requirements.txt

then:

    python pixelants.py

