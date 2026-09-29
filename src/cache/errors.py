"""Typed cache failures used to drive explicit recovery policy."""


class CacheError(RuntimeError):
    """Base class for cache protocol failures."""


class CacheNotFoundError(CacheError, FileNotFoundError):
    """No cache artifact or manifest exists."""


class CacheStaleError(CacheError):
    """Cache is validly encoded but no longer matches current semantics."""


class CacheLegacyError(CacheStaleError):
    """A pre-manifest cache exists and must be rebuilt once."""


class CacheIncompleteError(CacheError):
    """A cache generation is missing one or more committed artifacts."""


class CacheCorruptError(CacheError):
    """Cache bytes or internal metadata fail integrity validation."""


class CacheCompatibilityError(CacheError):
    """Cache was produced by a format this runtime must not overwrite."""
