# AGENTS.md

This file governs how an AI coding agent must behave, read it fully before touching any file.

---

## Prime Directives

1. **Simplicity over cleverness.** This is a small production project. If two implementations solve the same problem, always choose the one with fewer moving parts.
2. **Never break the sequential path.** The original particle filter must remain runnable at all times. It is the correctness reference and the local debug fallback.
3. **One concern per file.** Do not combine actor logic, orchestration logic, and filter logic in the same module. Each new file has exactly one job.
4. **No premature abstractions.** Do not create base classes, registries, or plugin systems unless a task explicitly requires it. Flat is better than clever.
5. **Preserve existing interfaces.** `main.py` must not change its public behavior. Any new parameter or flag must be additive and opt-in.