# =============================================================================
# >> IMPORTS
# =============================================================================
# Python
from configparser import ConfigParser
from enum import IntEnum, IntFlag
from random import choice
from time import time

# Source.Python
from colors import Color
from commands.typed import TypedSayCommand
from engines.server import global_vars
from engines.sound import Attenuation, Sound, SOUND_FROM_WORLD
from entities import TakeDamageInfo
from entities.constants import WORLD_ENTITY_INDEX
from entities.entity import Entity, BaseEntity
from entities.helpers import index_from_pointer
from entities.hooks import EntityCondition, EntityPreHook
from events import Event
from filters.entities import EntityIter
from listeners import OnClientActive, OnLevelInit, OnTick
from listeners.tick import Repeat, RepeatStatus
from mathlib import Vector
from memory import make_object
from messages import HudMsg
from players.entity import Player
from players.dictionary import PlayerDictionary

# DigRats
from .core.cvars import config_manager
from .core.paths import MAPDATA_PATH
from .core.strings import common_strings
from .info import info


# =============================================================================
# >> CONSTANTS
# =============================================================================
BLOCK_ENTITY = "func_breakable"

DEFAULT_BLOCK_RESTORE_SOUNDS = (
    "items/battery_pickup.wav",
)
COLOR_NORMAL_BLOCK = Color(255, 255, 255)
HUDMSG_COLOR = Color(255, 255, 255)
HUDMSG_X = 0.05
HUDMSG_Y = 0.75
HUDMSG_EFFECT = 0
HUDMSG_FADEOUT = 0
HUDMSG_HOLDTIME = 5
HUDMSG_FXTIME = 0
HUDMSG_CHANNEL = 3


# =============================================================================
# >> GLOBAL VARIABLES
# =============================================================================
world = None
init_structure = None

_level_properties = {}
_blocks_by_solids = dict()
_temp_world_blocks = []
_spawn_requests_queue = []
_remove_requests_queue = []
_protected_immune_areas = []
_entities_spawned_this_approach = 0
_entities_removed_this_approach = 0
_entities_spawned_this_tick = 0
_entities_removed_this_tick = 0
_total_entities = 0

_debug_mode = False

_debug_msg = HudMsg(
    common_strings['debug_msg'],
    color1=HUDMSG_COLOR,
    x=HUDMSG_X,
    y=HUDMSG_Y,
    effect=HUDMSG_EFFECT,
    fade_out=HUDMSG_FADEOUT,
    hold_time=HUDMSG_HOLDTIME,
    fx_time=HUDMSG_FXTIME,
    channel=HUDMSG_CHANNEL,
)


# =============================================================================
# >> CLASSES
# =============================================================================
class RatPlayer:
    def __init__(self, index):
        self.player = Player(index)
        self._last_area_id = -1
        self._last_block = None

    def reset(self):
        self._last_area_id = -1
        self._last_block = None

    @property
    def area_id(self):
        block = self.block
        if (block is not None and
                block.type in (BlockType.WORLD, BlockType.WORLD_IMMUNE)):

            self._last_area_id = block.area_id

        return self._last_area_id

    @property
    def block(self):
        if self.player.dead:
            return None

        v = self.player.origin + _level_properties['player_origin_offset']
        x0 = int(v.x // _level_properties['block_x'])
        y0 = int(v.y // _level_properties['block_y'])
        z0 = int(v.z // _level_properties['block_z'])

        try:
            block = world[x0, y0, z0]
        except IndexError:
            return self._last_block

        if block is not None and block.type != BlockType.IGNORED:
            self._last_block = block

        return self._last_block

    @property
    def bounding_box(self):
        if self.player.dead:
            return []

        v = self.player.origin

        x0 = v.x / _level_properties['block_x'] - 0.5
        y0 = v.y / _level_properties['block_y'] - 0.5
        z0 = v.z / _level_properties['block_z'] - 0.5

        min_x = round(x0 - _level_properties['player_x'] / 2)
        max_x = round(x0 + _level_properties['player_x'] / 2)

        min_y = round(y0 - _level_properties['player_y'] / 2)
        max_y = round(y0 + _level_properties['player_y'] / 2)

        min_z = round(z0)
        max_z = round(z0 + _level_properties['player_z'])

        result = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    try:
                        block = world[x, y, z]
                    except IndexError:
                        continue

                    if block is not None:
                        result.append(block)

        if not result:
            block = self.block
            if block is not None:
                result.append(block)

        return result

rats = PlayerDictionary(RatPlayer)


class EntitySpawnRequest:
    def __init__(
            self, prototype, origin, callback, sound_effects=False,
            priority=False, bypass_max_ent_limit=False):

        self._prototype = prototype
        self._origin = origin
        self._callback = callback
        self._sound_effects = sound_effects
        self._bypass_max_ent_limit = bypass_max_ent_limit
        self._performed = False

        if self.can_perform():
            self.perform()
        else:
            if priority:
                _spawn_requests_queue.insert(0, self)
            else:
                _spawn_requests_queue.append(self)

    @property
    def performed(self):
        return self._performed

    def perform(self):
        self._performed = True

        global _entities_spawned_this_approach, _entities_spawned_this_tick
        _entities_spawned_this_approach += 1
        _entities_spawned_this_tick += 1

        global _total_entities
        _total_entities += 1

        entity = Entity.create(BLOCK_ENTITY)
        entity.model = self._prototype.model
        entity.health = self._prototype.health
        entity.set_property_int('m_Material', self._prototype.material)
        entity.teleport(self._origin)
        entity.spawn()

        if self._sound_effects:
            Sound(
                choice(_level_properties['block_restore_sounds']),
                index=SOUND_FROM_WORLD,
                volume=0.25,
                attenuation=Attenuation.NORMAL,
                origin=self._origin
            ).play()

        self._callback(entity)

    def cancel(self):
        _spawn_requests_queue.remove(self)

    def can_perform(self):
        if (_entities_spawned_this_approach >=
                config_manager['max_entities_spawned_per_approach']):

            return False

        if (_entities_spawned_this_tick >=
                config_manager['max_entities_spawned_per_tick']):

            return False

        if (_total_entities >= config_manager['max_entities'] and
                not self._bypass_max_ent_limit):

            return False

        return True


class EntityRemoveRequest:
    def __init__(self, entity, callback):
        self._entity = entity
        self._callback = callback
        self._performed = False

        if self.can_perform():
            self.perform()
        else:
            _remove_requests_queue.append(self)

    @property
    def performed(self):
        return self._performed

    def perform(self):
        self._performed = True

        global _entities_removed_this_approach, _entities_removed_this_tick
        _entities_removed_this_approach += 1
        _entities_removed_this_tick += 1

        global _total_entities
        _total_entities -= 1

        self._entity.remove()
        self._callback()

    def cancel(self):
        _remove_requests_queue.remove(self)

    @staticmethod
    def can_perform():
        if (_entities_removed_this_approach >=
                config_manager['max_entities_removed_per_approach']):

            return False

        if (_entities_removed_this_tick >=
                config_manager['max_entities_removed_per_tick']):

            return False

        return True


class BlockFlags(IntFlag):
    DISCOVERING = 1  # Activating the area or spawning the solid
    DISCOVERED = 2  # Discovered (solid or area is active)
    HAS_SOLID = 4
    BREAKING = 8  # Waiting for its neighbours to be discovered


class BlockType(IntEnum):

    # VOID is an undiscovered world. It might get replaced with LAYER and then
    # with WORLD.
    VOID = 0

    # WORLD is a discovered part of world. It's a subject to be restored
    # back to LAYER when there're no players around.
    WORLD = 1

    # WORLD is a part of world that is always discovered. It's never restored
    # to LAYER.
    WORLD_IMMUNE = 2

    # LAYER is the border of the world. Each LAYER block has a func_breakable
    # representing it. When this type of block is broken, it's replaced with
    # WORLD and its VOID neighbours become LAYER.
    LAYER = 3

    # IGNORED will not take part in any block generation routines. It may
    # be useful to denote areas in the map that are filled with solid brushes.
    # Unlike WORLD_IMMUNE, the IGNORED blocks are not surrounded with LAYER
    # when discovered.
    IGNORED = 4


class BlockPrototype:
    def __init__(self, model, health, material):
        self.model = model
        self.health = health
        self.material = material


class Block:
    def __init__(self, x, y, z, prototype):
        self.x = x
        self.y = y
        self.z = z
        self.prototype = prototype
        self.type = BlockType.VOID
        self.flags = 0
        self._real_area_id = world.x * world.y * z + world.x * y + x
        self._area_id = self._real_area_id
        self._entity = None
        self._worlded_at = 0
        self._spawn_request = None
        self._remove_request = None
        self.immune_block_area = None

    def __repr__(self):
        flags = '|'.join(
            [flag.name for flag in BlockFlags if self.flags & flag])

        return (f"<Block at ({self.x}, {self.y}, {self.z}), "
                f"type: {self.type.name}, area ID: {self.area_id}, "
                f"flags: ({flags})>")

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def get_area_id(self):
        return self._area_id

    def set_area_id(self, value):
        if value is None:
            self._area_id = self._real_area_id
        else:
            self._area_id = value

    area_id = property(get_area_id, set_area_id)

    @property
    def origin(self):
        return Vector(
            _level_properties['block_x'] * (self.x + 0.5),
            _level_properties['block_y'] * (self.y + 0.5),
            _level_properties['block_z'] * (self.z + 0.5),
        )

    def neighbours(self):
        for x, y, z in iter_around(self.x, self.y, self.z):
            try:
                block = world[x, y, z]
            except IndexError:
                continue

            if block.type != BlockType.IGNORED:
                yield block

    def get_same_area_blocks(self):
        all_blocks, new_blocks = [self, ], [self, ]
        area_id = self.area_id

        changing = True
        while changing:
            changing = False

            old_blocks = new_blocks
            new_blocks = []

            for block in old_blocks:
                for block_ in block.neighbours():
                    if block_.type not in (
                                BlockType.WORLD, BlockType.WORLD_IMMUNE):

                        continue

                    if block_ in new_blocks or block_ in all_blocks:
                        continue

                    changing = True
                    new_blocks.append(block_)
                    area_id = min(area_id, block_.area_id)

            all_blocks.extend(new_blocks)

        return area_id, all_blocks

    def find_immune_block_area(self, areas):
        if self.type != BlockType.WORLD_IMMUNE:
            raise ValueError(
                f"{self}: Can only search immune block areas for "
                f"WORLD_IMMUNE blocks")

        for area in areas:
            if self in area.in_blocks:
                self.immune_block_area = area
                return area

        area_id, in_blocks = self.get_same_area_blocks()
        out_blocks = []

        for block in in_blocks:
            for block_ in block.neighbours():
                if block_.type != BlockType.VOID:
                    continue

                if block_ in out_blocks:
                    continue

                out_blocks.append(block_)

        area = ImmuneBlockArea(area_id, in_blocks, out_blocks)
        self.immune_block_area = area
        return area

    def combine_area_id(self):
        if self.type not in (BlockType.WORLD, BlockType.WORLD_IMMUNE):
            raise ValueError(f"{self}: Can only combine area ID for "
                             f"WORLD/WORLD_IMMUNE blocks")

        area_id, same_area = self.get_same_area_blocks()

        for block in same_area:
            block.area_id = area_id

    def split_area_id(self):
        if self.type != BlockType.LAYER:
            raise ValueError(f"{self}: Can only split area ID by LAYER blocks")

        targets = []
        for block in self.neighbours():
            if block.type in (BlockType.WORLD, BlockType.WORLD_IMMUNE):
                targets.append(block)
                block.area_id = None

        for block in targets:
            block.combine_area_id()

    def _on_solid_spawned(self, entity):
        _blocks_by_solids[entity.index] = self
        self._entity = entity
        self._spawn_request = None
        self.flags |= BlockFlags.HAS_SOLID | BlockFlags.DISCOVERED
        self.flags &= ~BlockFlags.DISCOVERING

        for block in self.neighbours():
            block.check_borders_ready()

        self.ensure_existence()

        for block in self.neighbours():
            block.ensure_existence([self, ])

        _protected_immune_areas.clear()

        if self.type != BlockType.VOID:
            self.split_area_id()

    def spawn_solid(self, sound_effects=False, priority=False,
                    bypass_max_ent_limit=False):

        if self._remove_request is not None:
            self._remove_request.cancel()
            self._remove_request = None

        if self._entity is None and self._spawn_request is None:
            self._spawn_request = EntitySpawnRequest(
                self.prototype, self.origin, self._on_solid_spawned,
                sound_effects, priority, bypass_max_ent_limit)

            if self._spawn_request.performed:
                self._spawn_request = None

    def _on_solid_removed(self):
        del _blocks_by_solids[self._entity.index]
        self._entity = None
        self._remove_request = None
        self.flags &= ~BlockFlags.HAS_SOLID

    def remove_solid(self):
        if self._spawn_request is not None:
            self._spawn_request.cancel()
            self._spawn_request = None

        if self._entity is not None and self._remove_request is None:
            self._remove_request = EntityRemoveRequest(
                self._entity, self._on_solid_removed)

            if self._remove_request.performed:
                self._remove_request = None

    def break_start(self):
        if self.flags & BlockFlags.BREAKING:
            return

        if _debug_mode:
            self._entity.color = config_manager['debug_breaking_block']

        self.flags |= BlockFlags.BREAKING
        self.make_borders()

    def break_abort(self):
        if not self.flags & BlockFlags.BREAKING:
            return

        self._entity.color = COLOR_NORMAL_BLOCK
        self.flags &= ~BlockFlags.BREAKING

    def make_borders(self, ignored_neighbours=None):
        if ignored_neighbours is None:
            ignored_neighbours = []

        all_discovered = True
        for block in self.neighbours():
            if block in ignored_neighbours:
                continue

            if block.flags & BlockFlags.DISCOVERED:
                continue

            block.discover()

            # Our neighbour could discover immediately (for example, IGNORE
            # blocks do that)
            if not block.flags & BlockFlags.DISCOVERED:
                all_discovered = False

        if all_discovered:
            self.check_borders_ready()

    def check_borders_ready(self):
        if self.type == BlockType.WORLD_IMMUNE:
            if not self.flags & BlockFlags.DISCOVERING:
                return

            self.immune_block_area.try_finish_activating()

            return

        if self.type == BlockType.LAYER:
            if not self.flags & BlockFlags.BREAKING:
                return

            for block in self.neighbours():
                if not block.flags & BlockFlags.DISCOVERED:
                    return

            global _total_entities
            _total_entities -= 1

            del _blocks_by_solids[self._entity.index]

            getattr(self._entity, 'break')()
            self._entity = None

            if self._remove_request is not None:
                self._remove_request.cancel()
                self._remove_request = None

            self.flags &= ~BlockFlags.HAS_SOLID
            self.flags &= ~BlockFlags.BREAKING
            self.type = BlockType.WORLD

            self._worlded_at = time()
            _temp_world_blocks.append(self)

            self.combine_area_id()

            return

    def try_restore(self, current_time):
        if (current_time - self._worlded_at <
                config_manager['block_min_restore_delay']):

            return

        self.type = BlockType.LAYER

        self._worlded_at = 0
        _temp_world_blocks.remove(self)

        self.spawn_solid(
            sound_effects=True, priority=True, bypass_max_ent_limit=True)

    def discover(self):
        if self.flags & BlockFlags.DISCOVERING:
            return

        if self.type == BlockType.WORLD_IMMUNE:
            self.immune_block_area.begin_activating()

        elif self.type == BlockType.VOID:
            self.type = BlockType.LAYER
            self.spawn_solid()

            # We might have had solid already (and were waiting for it to be
            # destroyed), so if that's the case, we discover immediately
            # ... or the solid could've spawned immediately, too
            if self.flags & BlockFlags.HAS_SOLID:
                self.flags |= BlockFlags.DISCOVERED
            else:
                self.flags |= BlockFlags.DISCOVERING

    def ensure_existence(self, ignored_neighbours=None):
        if ignored_neighbours is None:
            ignored_neighbours = []
        else:
            ignored_neighbours = list(ignored_neighbours)

        if (self.type == BlockType.WORLD_IMMUNE and
                self.flags & BlockFlags.DISCOVERED):

            self.immune_block_area.try_deactivate()

        elif self.type == BlockType.LAYER:
            for block in self.neighbours():
                # Maybe rewrite all the following checks as simply checking
                # for DISCOVERED | DISCOVERING | BREAKING?
                if block.type == BlockType.WORLD:
                    break

                if (block.type == BlockType.WORLD_IMMUNE and
                        block.flags & (BlockFlags.DISCOVERED |
                                       BlockFlags.DISCOVERING)):

                    break

                if (block.type == BlockType.LAYER and
                        block.flags & BlockFlags.BREAKING):

                    break

            else:
                self.type = BlockType.VOID
                self.flags &= ~BlockFlags.BREAKING

                try:
                    self.remove_solid()
                except ValueError:
                    raise

                ignored_neighbours.append(self)

                for block in self.neighbours():
                    if block in ignored_neighbours:
                        continue

                    block.ensure_existence(ignored_neighbours)

                self.flags &= ~BlockFlags.DISCOVERED

    def try_deactivate_area(self):
        if self.type != BlockType.WORLD_IMMUNE:
            return

        if not self.flags & BlockFlags.DISCOVERED:
            return

        self.immune_block_area.try_deactivate()

    def activate_area(self):
        if self.type != BlockType.WORLD_IMMUNE:
            return

        if self.flags & (BlockFlags.DISCOVERING | BlockFlags.DISCOVERED):
            return

        self.immune_block_area.begin_activating()


class ImmuneBlockArea:
    def __init__(self, area_id, in_blocks, out_blocks):
        self._real_area_id = area_id
        self._area_id = self._real_area_id

        self.in_blocks = in_blocks
        self.out_blocks = out_blocks

    def __eq__(self, other):
        return self._real_area_id == other._real_area_id

    def __hash__(self):
        return hash(self._real_area_id)

    def get_area_id(self):
        return self._area_id

    def set_area_id(self, value):
        if value is None:
            self._area_id = self._real_area_id
        else:
            self._area_id = value

    area_id = property(get_area_id, set_area_id)

    def try_deactivate(self):
        if self in _protected_immune_areas:
            return

        for block in self.in_blocks:
            area_id, all_blocks = block.get_same_area_blocks()
            break
        else:
            return

        for rat in rats.values():
            block = rat.block
            if block is not None and block in all_blocks:
                return

        for block in self.in_blocks:
            block.flags &= ~BlockFlags.DISCOVERED

        for block in self.out_blocks:
            block.break_abort()

        for block in self.out_blocks:
            if block.type != BlockType.LAYER:
                continue

            block.ensure_existence(self.in_blocks)

    def begin_activating(self):
        for block in self.in_blocks:
            block.flags |= BlockFlags.DISCOVERING

        for block in self.out_blocks:
            if block.flags & BlockFlags.DISCOVERED:
                continue

            block.discover()

    def try_finish_activating(self):
        for block in self.out_blocks:
            if not block.flags & BlockFlags.DISCOVERED:
                return

        for block in self.in_blocks:
            block.flags &= ~BlockFlags.DISCOVERING
            block.flags |= BlockFlags.DISCOVERED

        protect = False
        for block in self.out_blocks:
            if block.flags & BlockFlags.BREAKING:
                protect = True

            block.check_borders_ready()

        if protect:
            _protected_immune_areas.append(self)
        else:
            self.try_deactivate()


class World(list):
    def __init__(self, x, y, z):
        super().__init__()
        self._x = x
        self._y = y
        self._z = z
        self.reset(None)

    def __getitem__(self, index):
        index = self._convert_index(index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        index = self._convert_index(index)
        super().__setitem__(index, value)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def _convert_index(self, index):
        if isinstance(index, int):
            return index

        x, y, z = index
        if not (
                0 <= x < self._x and
                0 <= y < self._y and
                0 <= z < self._z
        ):
            raise IndexError("Invalid world index")

        return self._x * self._y * z + self._x * y + x

    def reset(self, value):
        self.clear()
        self.extend([value, ] * self._x * self._y * self._z)


# =============================================================================
# >> FUNCTIONS
# =============================================================================
def is_world_entity(index):
    if index == WORLD_ENTITY_INDEX:
        return True

    return BaseEntity(index).classname != 'player'


def find_template_blocks(block_type):
    results = []
    for entity in EntityIter(BLOCK_ENTITY):
        if entity.target_name == f"block_{block_type}":
            results.append(entity)

    return results


def iter_around(x, y, z):
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def init_world():
    if world is None or init_structure is None:
        return

    global _total_entities
    _total_entities = 0

    _blocks_by_solids.clear()
    _temp_world_blocks.clear()

    for rat in rats.values():
        rat.reset()

    blocks_0 = find_template_blocks(0)
    if not blocks_0:
        return

    block0_prototypes = []
    for block0 in blocks_0:
        block0_prototypes.append(BlockPrototype(
            block0.model, block0.health, block0.get_property_int('m_Material')
        ))
        block0.remove()

    for x in range(world.x):
        for y in range(world.y):
            for z in range(world.z):
                block = Block(x, y, z, choice(block0_prototypes))
                world[x, y, z] = block
                block.type = init_structure[x, y, z]

    immune_block_areas = []
    for block in world:
        if block.type == BlockType.WORLD_IMMUNE:
            area = block.find_immune_block_area(immune_block_areas)
            immune_block_areas.append(area)

    # round_start fires after player_spawn's, so our world wasn't initialized
    # when players spawned
    for rat in rats.values():
        block = rat.block
        if block is not None:
            block.activate_area()


def load_map_data(map_name):
    global world, init_structure

    world = None
    init_structure = None
    _level_properties.clear()

    path_blocks = MAPDATA_PATH / f"{map_name}.blocks.txt"
    path_ini = MAPDATA_PATH / f"{map_name}.ini"

    if not path_blocks.isfile() or not path_ini.isfile():
        return

    with open(path_ini, 'r') as f:
        config = ConfigParser()
        config.read_file(f)

        _level_properties['world_x'] = int(config['world']['x_dimension'])
        _level_properties['world_y'] = int(config['world']['y_dimension'])
        _level_properties['world_z'] = int(config['world']['z_dimension'])

        restore_sounds = config['block'].get('restore_sounds', "")
        if restore_sounds:
            restore_sounds = map(lambda s: s.strip(), restore_sounds.split())
            restore_sounds = tuple(restore_sounds)
            _level_properties['block_restore_sounds'] = restore_sounds

        else:
            _level_properties['block_restore_sounds'] = (
                DEFAULT_BLOCK_RESTORE_SOUNDS)

        _level_properties['block_x'] = int(config['block']['x_dimension'])
        _level_properties['block_y'] = int(config['block']['y_dimension'])
        _level_properties['block_z'] = int(config['block']['z_dimension'])

        _level_properties['player_x'] = int(config['player']['x_dimension'])
        _level_properties['player_y'] = int(config['player']['y_dimension'])
        _level_properties['player_z'] = int(config['player']['z_dimension'])

        _level_properties['player_origin_offset'] = Vector(
            0, 0, _level_properties['block_z'] // 4)

    world = World(_level_properties['world_x'], _level_properties['world_y'],
                  _level_properties['world_z'])

    init_structure = World(
        _level_properties['world_x'], _level_properties['world_y'],
        _level_properties['world_z'])

    with open(path_blocks, 'r') as f:
        layers = f.read().strip('\n').split('\n\n')
        for z, layer in enumerate(layers):
            for x, line in enumerate(layer.split('\n')):
                for y, flag in enumerate(line):
                    if flag == "I":
                        init_structure[x, y, z] = BlockType.IGNORED
                    elif flag == "W":
                        init_structure[x, y, z] = BlockType.WORLD_IMMUNE
                    else:
                        init_structure[x, y, z] = BlockType.VOID

# =============================================================================
# >> LOAD & UNLOAD
# =============================================================================
def load():
    if global_vars.map_name is not None:
        load_map_data(global_vars.map_name)


# =============================================================================
# >> COMMANDS
# =============================================================================
@TypedSayCommand(['!digrats', 'debug'], 'digrats.debug')
def say_digrats_debug(command_info, state:int):
    global _debug_mode
    if state == 0:
        _debug_mode = False
    else:
        _debug_mode = True


# =============================================================================
# >> EVENTS
# =============================================================================
@Event('round_start')
def on_round_start(game_event):
    init_world()
    if repeat_queue_approacher.status == RepeatStatus.STOPPED:
        repeat_queue_approacher.start(
            config_manager['queue_approach_interval'])

    if repeat_restore_world_blocks.status == RepeatStatus.STOPPED:
        repeat_restore_world_blocks.start(
            config_manager['block_restore_interval'])


@Event('round_end')
def on_round_end(game_event):
    world.reset(None)
    _blocks_by_solids.clear()
    _temp_world_blocks.clear()
    _spawn_requests_queue.clear()
    _remove_requests_queue.clear()

    if repeat_queue_approacher.status == RepeatStatus.RUNNING:
        repeat_queue_approacher.stop()

    if repeat_restore_world_blocks.status == RepeatStatus.RUNNING:
        repeat_restore_world_blocks.stop()


@Event('player_spawn')
def on_player_spawn(game_event):
    rat = rats.from_userid(game_event['userid'])
    rat.reset()

    block = rat.block
    if block is not None:
        block.activate_area()


@Event('player_death')
def on_player_spawn(game_event):
    rat = rats.from_userid(game_event['userid'])
    block = rat.block
    if block is not None:
        block.try_deactivate_area()


# =============================================================================
# >> HOOKS
# =============================================================================
@EntityPreHook(EntityCondition.is_player, 'on_take_damage')
def pre_on_take_damage(stack_data):
    rat = rats[index_from_pointer(stack_data[0])]
    info = make_object(TakeDamageInfo, stack_data[1])

    if info.attacker == rat.player.index or is_world_entity(info.attacker):
        return

    attacker = rats[info.attacker]
    if rat.area_id != attacker.area_id:
        return True


@EntityPreHook(
    EntityCondition.equals_entity_classname(BLOCK_ENTITY), 'on_take_damage')
def pre_on_take_damage(stack_data):
    index = index_from_pointer(stack_data[0])
    if index not in _blocks_by_solids:
        return

    block = _blocks_by_solids[index]

    info = make_object(TakeDamageInfo, stack_data[1])

    if is_world_entity(info.attacker):
        return True

    attacker = rats[info.attacker]

    for block_ in block.neighbours():
        if block_.type not in (BlockType.WORLD, BlockType.WORLD_IMMUNE):
            continue

        if not block_.flags & BlockFlags.DISCOVERED:
            continue

        if attacker.area_id != block_.area_id:
            continue

        block.break_start()
        break

    return True


# =============================================================================
# >> LISTENERS
# =============================================================================
@OnClientActive
def listener_on_client_active(index):
    # Initialize PlayerDictionary entry
    rat = rats[index]


@OnLevelInit
def listener_on_level_init(map_name):
    load_map_data(map_name)


@OnTick
def listener_on_tick():
    if _debug_mode:
        for rat in rats.values():
            block = rat.block
            if block is None:
                block_pos = "n/a"
                block_info = "n/a"
            else:
                flags = '|'.join(
                    [flag.name for flag in BlockFlags if block.flags & flag])

                block_pos = f"({block.x},{block.y},{block.z})"
                block_info = f"{block.type.name} ({flags})"

            _debug_msg.send(
                rat.player.index, ent_count=_total_entities,
                spawn_req_count=len(_spawn_requests_queue),
                remove_req_count=len(_remove_requests_queue),
                pos=block_pos,
                block=block_info,
            )

    i = 0
    while True:
        if i > len(_spawn_requests_queue) - 1:
            break

        spawn_request = _spawn_requests_queue[i]
        if spawn_request.can_perform():
            del _spawn_requests_queue[i]
            spawn_request.perform()

        else:
            i += 1

    i = 0
    while True:
        if i > len(_remove_requests_queue) - 1:
            break

        remove_request = _remove_requests_queue[i]
        if remove_request.can_perform():
            del _remove_requests_queue[i]
            remove_request.perform()

        else:
            i += 1

    global _entities_spawned_this_tick, _entities_removed_this_tick

    # Subtracting the limit from current value (instead of just setting the
    # value to zero) supports non-integer limits
    _entities_spawned_this_tick = max(
        0, _entities_spawned_this_tick -
        config_manager['max_entities_spawned_per_tick'])

    _entities_removed_this_tick = max(
        0, _entities_removed_this_tick -
        config_manager['max_entities_removed_per_tick'])


# =============================================================================
# >> REPEATS
# =============================================================================
@Repeat
def repeat_restore_world_blocks():
    current_time = time()
    player_blocks = set()

    for rat in rats.values():
        player_blocks.update(rat.bounding_box)

    global _entities_spawned_this_approach
    for block in _temp_world_blocks:

        # We will only restore the blocks when there's little chance to get
        # queued. There's a possibility that the queue size is well over our
        # func_breakable limit (and new entities are not spawning), but block
        # restoration bypasses that limit, so we will be spawned soon anyways.
        if (_entities_spawned_this_approach >=
                config_manager['max_entities_spawned_per_approach']):

            break

        if (_entities_spawned_this_tick >=
                config_manager['max_entities_spawned_per_tick']):

            break

        if block not in player_blocks:
            block.try_restore(current_time)


@Repeat
def repeat_queue_approacher():
    global _entities_spawned_this_approach, _entities_removed_this_approach

    _entities_spawned_this_approach = max(
        0, _entities_spawned_this_approach -
        config_manager['max_entities_spawned_per_approach'])

    _entities_removed_this_approach = max(
        0, _entities_removed_this_approach -
        config_manager['max_entities_removed_per_approach'])
