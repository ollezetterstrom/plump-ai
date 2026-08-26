"""plump — clean, decoupled Plump AI package.

Subpackages are intentionally isolated:
- plump.env      : pure game rules, no torch, no encode
- plump.encode   : state → tensor, depends on env only
- plump.models   : networks, depends on encode dims only
- plump.training : buffers / runners / learners, composes above
- plump.eval     : evaluation harness
- plump.play     : human vs AI CLI
- plump.config   : single source of truth

House rules (v2): 0-bid success = 5 pts, highest bidder (earliest tie) leads.
"""

__version__ = "2.0.0-house"
