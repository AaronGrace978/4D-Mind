"""Sentences grounded in a block's measured invariants.

The LoRA targets are these lines. A resident that has only the soft prompts
has to recover the numbers from the solid, because each life says something else.
"""

from __future__ import annotations

SYSTEM = (
    "You are a resident of the fourth dimension. A life is one unchanging block: "
    "every side, the interior, and every moment are coordinates you can see at once. "
    "Answer from the block you have been given. Do not narrate a sequence of befores and afters."
)

QUERY_TEXT = {
    "describe": "Describe this life as one solid.",
    "scar": "Where is the scar in the block?",
    "meeting": "Where do the two worldlines meet?",
    "interior": "What is inside the body, across the whole solid?",
}


def _inv(meta: dict) -> list[float]:
    return [float(value) for value in meta["invariants"]]


def scar_line(meta: dict) -> str:
    inv = _inv(meta)
    name = meta["name"]
    if inv[1] < 0:
        return f"No scar filament runs through {name}."
    return (
        f"A scar is a filament in {name} from t={inv[1]:.2f} through the end of the tube, "
        f"fixed at x={inv[3]:.2f}, y={inv[4]:.2f}, z={inv[5]:.2f}."
    )


def meeting_line(meta: dict) -> str:
    inv = _inv(meta)
    name = meta["name"]
    other = meta.get("other_name", "the other")
    if inv[2] < 0:
        return f"{other} does not touch {name} inside the visible solid."
    return (
        f"{other} touches {name} at t={inv[2]:.2f}, "
        f"x={inv[6]:.2f}, y={inv[7]:.2f}, z={inv[8]:.2f}. "
        f"The meeting is a joint in the solid."
    )


def interior_line(meta: dict) -> str:
    name = meta["name"]
    return (
        f"Inside {name}, tissue sits in the torso at every living slice. "
        f"It does not lie on the skin. Affect is a stain through that interior."
    )


def describe(meta: dict) -> str:
    inv = _inv(meta)
    name = meta["name"]
    parts = [
        f"{name} is one solid. The tube occupies {inv[0]:.0%} of the time axis.",
    ]
    if meta.get("same_room"):
        room = meta.get("room", [0.5, 0.5, 0.5])
        parts.append(
            "The childhood room and the deathbed are the same place, "
            f"x={float(room[0]):.2f} y={float(room[1]):.2f} z={float(room[2]):.2f}. "
            "The ends meet in space and differ only along t."
        )
    parts.append(scar_line(meta))
    parts.append(meeting_line(meta))
    parts.append(f"The body's spread through the block is {inv[9]:.2f}.")
    story = meta.get("story") or ""
    if story:
        parts.append(story)
    return "\n\n".join(parts)


def answer_for(meta: dict, key: str) -> str:
    if key == "describe":
        return describe(meta)
    if key == "scar":
        return scar_line(meta)
    if key == "meeting":
        return meeting_line(meta)
    if key == "interior":
        return interior_line(meta)
    raise KeyError(key)


def probe_sentence(channels: dict[str, float], x: float, y: float, z: float, t: float) -> str:
    bits = [f"Point x={x:.2f} y={y:.2f} z={z:.2f} t={t:.2f}."]
    if channels["occupancy"] > 0.3:
        bits.append("This coordinate is inside the body.")
    else:
        bits.append("This coordinate is outside the body.")
    if channels["tissue"] > 0.3:
        bits.append("Interior tissue is here, hidden from the surface.")
    if channels["scar"] > 0.3:
        bits.append("The scar filament passes through this point.")
    if channels["other"] > 0.3:
        bits.append("The second worldline occupies this point.")
    bits.append(f"Affect stained on the solid: {channels['affect']:.2f}.")
    return " ".join(bits)
