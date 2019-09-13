"""
Instead of adding six as a dependency, this code was copied from the six implementation.
six is Copyright (c) 2010-2015 Benjamin Peterson
"""


# TODO: Perhaps rename this to platform.py or something and add OS-specific hardware detection.

def iterkeys(d, **kw):
    return iter(d.keys(**kw))

def iteritems(d, **kw):
    return iter(d.items(**kw))

def itervalues(d, **kw):
    return iter(d.values(**kw))
