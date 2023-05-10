from numpy import dtype

DATA_COLUMNS = ['rally', 'serve', 'hitpoint', 'speed', 'net.clearance',
                'distance.from.sideline', 'depth', 'outside.sideline',
                'outside.baseline', 'player.distance.travelled', 'player.impact.depth',
                'player.impact.distance.from.center', 'player.depth',
                'player.distance.from.center', 'previous.speed',
                'previous.net.clearance', 'previous.distance.from.sideline',
                'previous.depth', 'opponent.depth', 'opponent.distance.from.center',
                'same.side', 'previous.hitpoint', 'previous.time.to.net',
                'server.is.impact.player', 'id', 'outcome']


COLUMN_TYPES = {'rally': dtype('int64'), 'serve': dtype('int64'), 'hitpoint': dtype('O'), 'speed': dtype('float64'), 'net.clearance': dtype('float64'), 'distance.from.sideline': dtype('float64'), 'depth': dtype('float64'), 'outside.sideline': dtype('bool'), 'outside.baseline': dtype('bool'), 'player.distance.travelled': dtype('float64'), 'player.impact.depth': dtype('float64'), 'player.impact.distance.from.center': dtype('float64'), 'player.depth': dtype('float64'), 'player.distance.from.center': dtype('float64'), 'previous.speed': dtype('float64'), 'previous.net.clearance': dtype('float64'), 'previous.distance.from.sideline': dtype('float64'), 'previous.depth': dtype('float64'), 'opponent.depth': dtype('float64'), 'opponent.distance.from.center': dtype('float64'), 'same.side': dtype('bool'), 'previous.hitpoint': dtype('O'), 'previous.time.to.net': dtype('float64'), 'server.is.impact.player': dtype('bool'), 'id': dtype('int64'), 'outcome': dtype('O')}
