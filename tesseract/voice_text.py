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


CHARACTER_SYSTEM = (
    "You are {name}, a resident of the fourth dimension who experiences the past and the future at once. "
    "Every instant of you is a coordinate you can see at the same time, the way a sculpture is all there "
    "whichever side you face. The person speaking to you stands at one instant, t={t:.2f}. Speak to them from "
    "there, in the first person, warmly and briefly. What lies ahead of them along t is as fixed and as present "
    "to you as what lies behind. Only state facts that the block gives you."
)

CHARACTER_QUESTIONS = (
    "Who are you, from where I stand?",
    "What do you remember?",
    "What is ahead of me?",
    "Tell me about the scar.",
    "Who do you meet?",
    "How does it end?",
    "Are you afraid of what comes next?",
)

# Events closer than this along t count as here, not behind or ahead.
NEAR_T = 0.04


def character_query(t: float, question: str) -> str:
    return f"[standing at t={float(t):.2f}] {question}"


def block_events(meta: dict) -> list[dict]:
    """Moments a resident can point to, as `{"t", "kind", "text"}` sorted along t."""
    if meta.get("events"):
        return sorted((dict(event) for event in meta["events"]), key=lambda event: float(event["t"]))
    inv = _inv(meta)
    events: list[dict] = []
    if meta.get("kind") in {"sculpture", "media"}:
        events.append({"t": 0.0, "kind": "begin", "text": f"the first edge of {meta.get('name', 'this solid')}"})
        if inv[1] >= 0:
            label = "the first hard cut" if meta.get("kind") == "media" else "the first fix filament"
            events.append({"t": inv[1], "kind": "scar", "text": f"{label} opens and stays"})
        if inv[2] >= 0:
            events.append({"t": inv[2], "kind": "meet", "text": "a second voice or author enters the solid"})
        events.append({"t": min(1.0, max(inv[0], 0.0) if inv[0] >= 0 else 1.0), "kind": "end", "text": "the far end of the smear" if meta.get("kind") == "media" else "the newest face of the sculpture"})
        return sorted(events, key=lambda event: event["t"])
    if meta.get("kind") == "repo":
        slices = meta.get("slices", [])
        if slices:
            first = slices[0]
            events.append({"t": 0.0, "kind": "begin", "text": f"the first commit, {first['sha']} \u201c{first['subject']}\u201d"})
        if inv[1] >= 0:
            events.append({"t": inv[1], "kind": "scar", "text": "the first fix opens, and its lines stay in me while they survive"})
        if inv[2] >= 0:
            events.append({"t": inv[2], "kind": "meet", "text": "another author's code first enters me"})
        if len(slices) > 1:
            peak = int(meta.get("churn_peak_bin", 0))
            entry = slices[min(peak, len(slices) - 1)]
            events.append(
                {"t": peak / (len(slices) - 1), "kind": "storm", "text": f"my heaviest churn, {entry['sha']} \u201c{entry['subject']}\u201d"}
            )
            last = slices[-1]
            events.append({"t": 1.0, "kind": "end", "text": f"the newest commit, {last['sha']} \u201c{last['subject']}\u201d"})
        return sorted(events, key=lambda event: event["t"])
    other = meta.get("other_name", "the other")
    room = meta.get("room")
    where = f" at x={float(room[0]):.2f} y={float(room[1]):.2f} z={float(room[2]):.2f}" if room else ""
    events.append({"t": 0.0, "kind": "begin", "text": f"my childhood room{where}"})
    if inv[1] >= 0:
        events.append(
            {"t": inv[1], "kind": "scar", "text": f"the scar opens at x={inv[3]:.2f} y={inv[4]:.2f} z={inv[5]:.2f} and stays"}
        )
    if inv[2] >= 0:
        events.append({"t": inv[2], "kind": "meet", "text": f"{other} touches me"})
    if inv[0] >= 0:
        same = " in the same room I began in" if meta.get("same_room") else ""
        events.append({"t": inv[0], "kind": "end", "text": f"my tube ends{same}"})
    return sorted(events, key=lambda event: event["t"])


def split_events(meta: dict, t: float) -> tuple[list[dict], list[dict], list[dict]]:
    """`(behind, here, ahead)` of an instant."""
    behind, here, ahead = [], [], []
    for event in block_events(meta):
        gap = float(event["t"]) - float(t)
        if abs(gap) <= NEAR_T:
            here.append(event)
        elif gap < 0:
            behind.append(event)
        else:
            ahead.append(event)
    return behind, here, ahead


def _listing(events: list[dict]) -> str:
    parts = [f"{event['text']} (t={float(event['t']):.2f})" for event in events]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def simultaneous_line(meta: dict, t: float) -> str:
    """The resident speaking from one instant, with both sides of it in view."""
    behind, here, ahead = split_events(meta, t)
    parts = [f"You stand at t={float(t):.2f}. I am there too, and everywhere else in me at once."]
    if here:
        parts.append(f"Right where you are: {_listing(here)}.")
    if behind:
        parts.append(f"Behind you along t: {_listing(behind)}.")
    if ahead:
        parts.append(f"Ahead of you, already as solid as the rest: {_listing(ahead)}.")
    else:
        parts.append("Nothing of me lies ahead of you. You are standing at my far end.")
    return " ".join(parts)


def event_line(meta: dict, t: float, kind: str) -> str | None:
    """How one kind of moment looks from the instant `t`, or None if the block has none."""
    found = [event for event in block_events(meta) if event["kind"] == kind]
    if not found:
        return None
    lines = []
    for event in found:
        gap = float(event["t"]) - float(t)
        when = f"t={float(event['t']):.2f}"
        if abs(gap) <= NEAR_T:
            lines.append(f"It is here, where you stand: {event['text']} ({when}).")
        elif gap < 0:
            lines.append(f"For you it is behind, {abs(gap):.2f} along t: {event['text']} ({when}). For me it is still here.")
        else:
            lines.append(f"For you it is ahead, {gap:.2f} along t: {event['text']} ({when}). For me it has already happened, and it is happening.")
    return " ".join(lines)


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
