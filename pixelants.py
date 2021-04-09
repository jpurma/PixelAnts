import numpy as np
from random import random, randint, choice
import math

from kivy.app import App
from kivy.graphics.context_instructions import Color
from kivy.graphics.fbo import Fbo
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Line, Rectangle
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.core.window import Window

SIZE = W, H = [800, 800]
HIVE_CENTER = W / 2, H / 2
Window.size = SIZE

SPEED = 1.5
ANTS = 400
RETURN_AT = 2000
GOALS = 30
ROCKS = 30
DECAY_FREQ = 100
NEW_FOOD_FREQ = 200
GOAL_SIZE = (20, 150)
IN_SCENT_STRENGTH = 20
PATH_INTEGRATION = False

STONE_COLOR = [80, 80, 80, 255]
STONE_COLOR_FLOAT = list(np.array(STONE_COLOR) / 255)

ant_list = []

ant_world_array = np.zeros(SIZE + [4], dtype=np.uint8)
world_array = np.zeros(SIZE + [4], dtype=np.uint8)
food_array = np.zeros(SIZE, dtype=np.uint32)
scent_out_distance_array = np.zeros(SIZE, dtype=np.uint32)
scent_out_direction_array = np.zeros(SIZE, dtype=np.float32)
scent_in_decay_array = np.zeros(SIZE + [10], dtype=np.int32)
scent_in_direction_array = np.zeros(SIZE + [10], dtype=np.float32)
scent_in_decay_element_count_array = np.zeros(SIZE, dtype=np.uint32)
scent_in_rgba_array = np.zeros(SIZE + [4], dtype=np.uint8)
scent_in_rgba_base_array = np.full(SIZE + [4], [128, 128, 240, 0], dtype=np.uint8)
scent_in_alpha_base_array = np.full(SIZE + [4], [5, 7, 1, 20], dtype=np.uint8)

PI2 = math.pi * 2
MAX_X = W - 1
MAX_Y = H - 1


def limit_radian(r):
    if r < 0:
        return r + PI2
    elif r >= PI2:
        return r - PI2
    return r


def limit(min_v, value, max_v):
    return min((max_v, max((min_v, value))))


def rad_dist(r1, r2):
    if r1 > r2:
        if abs(r1 - r2) < abs(r1 - (r2 + PI2)):
            return r1 - r2
        else:
            return r1 - (r2 + PI2)
    if abs(r1 - r2) < abs(r1 + PI2 - r2):
        return r1 - r2
    return r1 + PI2 - r2


def randn_bm():
    u = 0
    v = 0
    while u == 0:
        u = random()
    while v == 0:
        v = random()
    num = math.sqrt(-2 * math.log(u)) * math.cos(PI2 * v)
    num = num / 10.0 + 0.5  # Translate to 0 -> 1
    if num > 1 or num < 0:
        num = randn_bm()  # resample between 0 and 1 if out of range
    return num


def d_change():
    return randn_bm() * PI2 - math.pi


def opposite(d):
    nd = d + math.pi
    return nd - PI2 if nd > PI2 else nd


def is_occupied(x, y):
    if y < 0 or y > MAX_Y or x < 0 or x > MAX_X:
        return True
    rgba = world_array[y][x]
    return all(rgba == STONE_COLOR)


class Ant:
    FORAGING = 0
    FORAGING_RGBA = [255, 225, 50, 255]
    RETURNING = 1
    RETURNING_RGBA = [230, 230, 255, 255]
    RETURNING_EMPTY = 2
    RETURNING_EMPTY_RGBA = [255, 0, 50, 255]

    def __init__(self, x, y):
        self.mode = Ant.FORAGING
        self.has_last_route = False
        self.last_route_dist = 0
        self.last_route_d = 0.0
        self.travel_time = 0
        self.turn_direction = -1 if random() > 0.5 else 1
        self.x = int(x)
        self.fx = float(x)
        self.y = int(y)
        self.fy = float(y)
        self.d = random() * PI2
        self.home_d = 0.0
        self.dist = 0.0

    def limit_to_area(self):
        # remain inside world limits
        hit = False
        if self.x < 0:
            self.fx = 0.0
            hit = True
        elif self.x > MAX_X:
            self.fx = float(MAX_X)
            hit = True
        if self.y < 0:
            self.fy = 0.0
            hit = True
        elif self.y > MAX_Y:
            self.fy = float(MAX_Y)
            hit = True
        if hit:
            self.x = int(self.fx)
            self.y = int(self.fy)
            self.d = opposite(self.d)

    def at_food(self):
        if food_array[self.y][self.x]:
            return True

    def at_home(self):
        if self.dist < 9:
            return True

    def leave_scent_out(self, as_opposite=False):
        scent_closeness = scent_out_distance_array[self.y][self.x]
        if not scent_closeness or scent_closeness > self.travel_time:
            scent_out_distance_array[self.y][self.x] = self.travel_time
            scent_out_direction_array[self.y][self.x] = self.d if not as_opposite else opposite(self.d)
            closeness_log = math.log2(max((1, 5000 - self.travel_time))) / 13
            r, g, b = Color(self.d / PI2, 0.9, closeness_log, mode='hsv').rgb
            world_array[self.y][self.x] = [r * 255, g * 255, b * 255, 80]

    def leave_scent_in(self):
        decays = [decay for decay in scent_in_decay_array[self.y][self.x] if decay > 0]
        directions = list(scent_in_direction_array[self.y][self.x])
        directions = directions[:len(decays)]
        decays.append(IN_SCENT_STRENGTH)
        directions.append(self.d)
        scents_size = len(decays)
        if len(decays) > 10:
            decays = decays[1:]
            directions = directions[1:]
        elif len(decays) < 10:
            decays += [0] * (10 - len(decays))
            directions += [0] * (10 - len(directions))
        scent_in_decay_array[self.y][self.x] = decays
        scent_in_direction_array[self.y][self.x] = directions
        scent_in_decay_element_count_array[self.y][self.x] = scents_size
        scent_in_rgba_array[self.y][self.x] = [128 + 5 * scents_size,
                                               128 + 7 * scents_size,
                                               240 + scents_size,
                                               20 + 20 * scents_size]

    def pick_scent_out(self, chance=.3, rebel=.1):
        if random() < rebel or random() > chance:
            return
        if scent_out_distance_array[self.y][self.x]:
            return scent_out_direction_array[self.y][self.x]

    def pick_scent_in(self, chance=.3, rebel=.1):
        if random() < rebel:
            return
        scents = [i for i, decay in enumerate(scent_in_decay_array[self.y][self.x]) if decay > 0]
        for s in scents:
            if random() < chance:
                scent_direction = scent_in_direction_array[(self.y, self.x, choice(scents))]
                self.has_last_route = True
                self.last_route_dist = 0
                return scent_direction

    def compute_home(self, speed, target_rad, current_rad, current_dist):
        """ compute the angle and distance to home based on the previous angle to target and dist, and the angle and speed
         of current movement. This is optional, if we want to try what happens with path integration capability."""

        c = math.pi - (target_rad - current_rad)
        speed2 = speed * speed
        dist2 = current_dist * current_dist
        new_dist2 = speed2 + dist2 - 2 * speed * current_dist * math.cos(c)
        if current_dist <= 0 or new_dist2 <= 0:
            return current_rad, speed
        new_dist = math.sqrt(new_dist2)
        cos_rule = (dist2 + new_dist2 - speed2) / (2 * current_dist * new_dist)
        cos_rule = limit(-1, cos_rule, 1)  # floating point inaccuracies may lead to 1.0000000002
        diff = math.acos(cos_rule)
        if rad_dist(current_rad, target_rad) < 0:
            diff = -diff
        return limit_radian(target_rad + diff), new_dist

    def move_step(self, speed):
        attempts = 0
        while True:
            attempts += 1
            fx = self.fx + speed * math.cos(self.d)
            fy = self.fy + speed * math.sin(self.d)
            x = int(fx)
            y = int(fy)
            if not is_occupied(x, y):
                break
            if random() < 0.0001:
                self.turn_direction *= -1
                attempts = -attempts
            change = math.pi / 32 * self.turn_direction
            #change = abs(d_change() / 8) * self.turn_direction
            self.d = limit_radian(self.d + change)
            if attempts == 64:
                fx, fy = HIVE_CENTER
                x = int(fx)
                y = int(fy)
                print('dead end')
                break
        self.fx = fx
        self.fy = fy
        self.x = x
        self.y = y
        self.limit_to_area()
        if PATH_INTEGRATION:
            self.home_d, self.dist = self.compute_home(speed, self.home_d, self.d, self.dist)
        else:
            self.dist = math.dist(HIVE_CENTER, (self.x, self.y))
        if self.has_last_route:
            self.last_route_d, self.last_route_dist = self.compute_home(speed, self.last_route_d, self.d, self.last_route_dist)
        self.draw()

    def draw(self):
        if self.mode == Ant.RETURNING:
            color = Ant.RETURNING_RGBA
        elif self.mode == Ant.RETURNING_EMPTY:
            color = Ant.RETURNING_EMPTY_RGBA
        else:
            color = Ant.FORAGING_RGBA
        ant_world_array[self.y][self.x] = color

    def travel(self, speed):
        self.travel_time += 1
        if self.mode == Ant.FORAGING or Ant.RETURNING_EMPTY:
            if self.at_food():
                food_array[self.y][self.x] -= 1
                self.mode = Ant.RETURNING
                food_left = food_array[self.y][self.x]
                world_array[self.y][self.x] = [255, 160, 160, min((255, 200 + food_left * 2)) if food_left else 0]
                self.d = opposite(self.d)

        if self.mode == Ant.RETURNING:
            if (scent_out_direction := self.pick_scent_out(.7)) is not None:
                self.d = opposite(scent_out_direction)
            # elif (scent_in_direction := self.pick_scent_in(.7)) is not None:
            #     self.d = scent_in_direction
            #     self.has_last_route = True
            # elif PATH_INTEGRATION and self.has_last_route:
            #     self.d = opposite(self.last_route_d)
            #     if self.last_route_dist <= speed:
            #         self.has_last_route = False
            elif PATH_INTEGRATION and self.travel_time < 1000:
                # Go home
                self.d = opposite(self.home_d)
            else:
                # Random walk
                self.d = limit_radian(self.d + d_change() / 4)
            self.move_step(speed)
            if self.at_home():
                self.mode = Ant.FORAGING
                self.travel_time = 0
            else:
                self.leave_scent_in()

        elif self.mode == Ant.RETURNING_EMPTY:
            if (scent_out_direction := self.pick_scent_out(.7)) is not None:
                self.d = opposite(scent_out_direction)
            else:
                # Random walk
                self.d = limit_radian(self.d + d_change() / 4)
            self.move_step(speed)
            if self.at_home():
                self.mode = Ant.FORAGING
                self.travel_time = 0

        else:
            if (scent_in_direction := self.pick_scent_in(.7)) is not None:
                self.d = opposite(scent_in_direction)
                self.has_last_route = True
            else:
                # Random walk
                self.d = limit_radian(self.d + d_change() / 4)
            self.move_step(speed)
            self.leave_scent_out()
            if self.travel_time > RETURN_AT:
                self.mode = Ant.RETURNING_EMPTY


class Hive:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.load = 0


class Rock:
    def __init__(self):
        point_n = randint(6, 12)
        radius = randint(20, 100)
        screen_cx = MAX_X / 2
        screen_cy = MAX_Y / 2
        center_x = screen_cx
        center_y = screen_cy
        while abs(screen_cx - center_x) < radius and abs(screen_cy - center_y) < radius:
            center_x = randint(0, MAX_X)
            center_y = randint(0, MAX_Y)
        self.points = []
        rotation = randint(0, 12)
        for n in range(rotation, point_n + rotation):
            x_adj = int(random() * (radius / 2) - radius / 4)
            y_adj = int(random() * (radius / 2) - radius / 4)
            rad = n * (PI2 / (point_n + 6))
            x = center_x + math.cos(rad) * radius + x_adj
            y = center_y + math.sin(rad) * radius + y_adj
            self.points.append(x)
            self.points.append(y)


def turn_to_world_array(pixels):
    global world_array
    buffer = np.frombuffer(pixels, dtype=np.uint8)
    world_array = np.reshape(buffer, (W, H, 4)).copy()


class World(Widget):
    def __init__(self, app):
        super(World, self).__init__()
        self.size_hint = None, None
        self.size = SIZE
        self.pos_hint = {'center': (.5, 5)}
        self.hive = None
        self.rocks = []
        self.add_ground()
        w, h = SIZE
        with self.canvas:
            ground_fbo = Fbo(size=self.size)
            Rectangle(pos=(0, 0), size=(w * 2, h * 2), texture=ground_fbo.texture)
            Rectangle(pos=(0, 0), size=(w * 2, h * 2), texture=app.texture)
            Rectangle(pos=(0, 0), size=(w * 2, h * 2), texture=app.paths_in_texture)
        with ground_fbo:
            Color(*STONE_COLOR_FLOAT)
            Line(rectangle=[0, 0, w, h], width=4)
            for rock in self.rocks:
                Line(points=rock.points, close=False, width=4)
            ground_fbo.draw()
        turn_to_world_array(ground_fbo.texture.pixels)
        self.populate()

    def add_food(self):
        c_x = randint(0, MAX_X)
        c_y = randint(0, MAX_Y)
        load = randint(*GOAL_SIZE)
        size = load / 3
        for i in range(0, load):
            x = int(round(limit(0, c_x + (random() * load / 2.5) - size, MAX_X)))
            y = int(round(limit(0, c_y + (random() * load / 2.5) - size, MAX_Y)))
            food_array[y][x] += 1
            world_array[y][x] = [255, 160, 160, min((255, 200 + food_array[y][x] * 2))]

    def global_decay(self):
        global scent_in_decay_element_count_array, scent_in_rgba_array
        np.putmask(scent_in_decay_array, scent_in_decay_array >= 1, scent_in_decay_array - 1)
        scent_in_decay_element_count_array = np.array(np.ma.count_masked(
            np.ma.array(
                scent_in_decay_array,
                mask=np.ma.make_mask(scent_in_decay_array),
                dtype=np.uint8
            ), 2),
            dtype=np.uint8)
        scent_in_alpha_array = scent_in_alpha_base_array * scent_in_decay_element_count_array.reshape(W, H, 1)
        scent_in_rgba_array = scent_in_rgba_base_array + scent_in_alpha_array

    def add_ground(self):
        for rock_id in range(0, ROCKS):
            rock = Rock()
            self.rocks.append(rock)

        x, y = HIVE_CENTER
        hive = Hive(x=x, y=y)
        self.hive = hive

    def populate(self):
        x = self.hive.x
        y = self.hive.y
        for ant_id in range(0, ANTS):
            ant = Ant(x, y)
            ant_list.append(ant)

        for goal_id in range(0, GOALS):
            self.add_food()


class PixelAnts(App):

    def __init__(self, **kwargs):
        super(PixelAnts, self).__init__(**kwargs)
        self.main = None
        self.tick = 0
        self.paths_in_texture = None

    def build(self):
        Clock.schedule_interval(self.refresh, 0)
        self.texture = Texture.create(size=SIZE, colorfmt='rgba', bufferfmt='ubyte')
        self.texture.min_filter = 'nearest'
        self.texture.mag_filter = 'nearest'
        self.paths_in_texture = Texture.create(size=SIZE, colorfmt='rgba', bufferfmt='ubyte')
        self.paths_in_texture.min_filter = 'nearest'
        self.paths_in_texture.mag_filter = 'nearest'
        self.main = World(self)
        return self.main

    def refresh(self, dt):
        global ant_world_array
        ant_world_array = world_array.copy()
        self.tick += 1
        if self.tick % DECAY_FREQ == 0:
            self.main.global_decay()
        if self.tick % NEW_FOOD_FREQ == 0:
            self.main.add_food()
        for ant in ant_list:
            ant.travel(SPEED)
        self.texture.blit_buffer(ant_world_array.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.paths_in_texture.blit_buffer(scent_in_rgba_array.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.main.canvas.flag_update()


if __name__ == '__main__':
    PixelAnts().run()