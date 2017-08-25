# DigRats

## Introduction
**DigRats** is a Source.Python plugin that drastically reduces the concurrent amount of breakable entities on breakfloor-like maps.

It keeps only those blocks that are directly visible to players.

Once a player leaves an area, blocks start restoring behind them - but in fact even more blocks are removed because they've been hidden by the restored ones.

The principle can be seen in this video: https://youtu.be/U3bTLqXQkVQ

## Ideas
Besides the main idea, there're few others that help keeping the server healthy.

### Spawning/Removing queues
If you break a regular block, it will usually take 5 another to spawn around it.

But if you break a block that hides some pre-defined open area (like in the above video), you'll need to spawn the whole bordering layer of blocks for that area.

Trying to do so immediately will simply freeze the server for a few seconds. Thus the spawning and removing queues were introduced.

They limit the number of blocks that are spawned (or removed) every tick, avoiding server stuttering.

This means that the block won't break right away, it will wait until the area hidden by it is fully built.

Some blocks may require a second or two to break, but nobody except for the shooter will notice this delay.

### Max entity limit
Even though the blocks are "restored" (in fact, for every restored block there're multiple removed), with increasing number of players some problems still may occur.

For example, block restoration rates may simply not keep up with the speed that players discover new blocks.

That can lead to some serious issues (hitting engine's [`MAX_EDICTS`](https://developer.valvesoftware.com/wiki/Entity_limit) limit or crashing clients),
thus a more strict feature is implemented - the limit of discovered blocks.

Once it's reached, new blocks are not created anymore (until the number of discovered blocks falls down).

To avoid deadlock situation, the block restoration routine will bypass this limitation.

This limit must be kept high enough so that the players are not trapped in their own base.

Classic ***breakfloor*** map from Counter-Strike was built using 860 blocks, so that's your reference value.

The ***breakfloor_4096*** map (first map utilizing the DigRats features) requires 422 blocks to surround its pre-defined open areas (including player spawn bases),
but you need to keep the limit above this minimum so that players could get out of those areas.

Current default value of `digrats_max_entities` is 500.

### Tracing enemies by block color (unimplemented)
One good thing about the original breakfloor maps is that you don't only dig your own tunnels - you can search the tunnels made by the opposite team!

Unfortunately, the restoring routine of this plugin makes this impossible...
Or not? The idea is to colorize the restored blocks with the color of the team that previously broken it.

## Maps
Every map for this plugin consists of the `.bsp` file (the map itself) and two data files: `<map name>.blocks.txt` and `<map name>.ini`.
The blocks file defines the layout of open areas while the `ini` file stores general world information.

### breakfloor_4096
Currently the only map for this plugin is made by me. Its world is 16x16x16 (thus 4096 in the title), but the actual number of breakable blocks is 3474.
The `.bsp` can be downloaded [here](https://yadi.sk/d/D0fMc6K13MKtua), the data files for this map are shipped with DigRats ([/srcds/mapdata/digrats](https://github.com/KirillMysnik/SP-DigRats/tree/master/srcds/mapdata/digrats)).
