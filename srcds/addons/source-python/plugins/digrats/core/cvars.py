# =============================================================================
# >> IMPORTS
# =============================================================================
# Custom Package
from controlled_cvars import ControlledConfigManager, InvalidValue
from controlled_cvars.handlers import color_handler, float_handler, int_handler

# Map Cycle
from ..info import info
from .strings import config_strings


def uint_handler(cvar):
    value = int_handler(cvar)
    if value < 0:
        raise InvalidValue
    return value


def ufloat_handler(cvar):
    value = float_handler(cvar)
    if value < 0:
        raise InvalidValue
    return value


config_manager = ControlledConfigManager(info.name, cvar_prefix='digrats_')

config_manager.section(config_strings['section debug_settings'])
cvar_timelimit = config_manager.controlled_cvar(
    color_handler,
    "debug_breaking_block",
    default="255,0,0,50",
    description=config_strings['debug_breaking_block'],
)

config_manager.section(config_strings['section block_settings'])
config_manager.controlled_cvar(
    ufloat_handler,
    name="block_restore_interval",
    default=0.2,
    description=config_strings['block_restore_interval'],
)
config_manager.controlled_cvar(
    ufloat_handler,
    name="block_min_restore_delay",
    default=2.0,
    description=config_strings['block_min_restore_delay'],
)

config_manager.section(config_strings['section limit_settings'])
config_manager.controlled_cvar(
    ufloat_handler,
    name="max_entities",
    default=500,
    description=config_strings['max_entities'],
)
config_manager.controlled_cvar(
    ufloat_handler,
    name="queue_approach_interval",
    default=1.0,
    description=config_strings['queue_approach_interval'],
)
config_manager.controlled_cvar(
    ufloat_handler,
    name="max_entities_spawned_per_tick",
    default=6,
    description=config_strings['max_entities_spawned_per_tick'],
)
config_manager.controlled_cvar(
    ufloat_handler,
    name="max_entities_removed_per_tick",
    default=6,
    description=config_strings['max_entities_removed_per_tick'],
)
config_manager.controlled_cvar(
    ufloat_handler,
    name="max_entities_spawned_per_approach",
    default=20,
    description=config_strings['max_entities_spawned_per_approach'],
)
config_manager.controlled_cvar(
    ufloat_handler,
    name="max_entities_removed_per_approach",
    default=20,
    description=config_strings['max_entities_removed_per_approach'],
)

config_manager.write()
config_manager.execute()
