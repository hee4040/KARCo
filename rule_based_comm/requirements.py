"""Scene-only task requirements, before considering B's knowledge."""

from .schemas import Action, Event, TaskRequirements


def determine_requirements(event: Event) -> TaskRequirements:
    if event.already_processing and not event.scene_changed:
        return TaskRequirements(
            status=Action.NO_COMM,
            reasons=("same event is already processing and the scene is unchanged",),
        )

    if not event.target_confirmed:
        return TaskRequirements(
            status=Action.REOBSERVE,
            reasons=("sender has not confirmed the target",),
        )

    required = {"E", "N"}
    reasons = ["event and normal pose are needed to complete restoration"]
    if event.has_similar_objects or event.similar_object_count > 0:
        required.update(("C", "I"))
        reasons.append("lookalikes require current pose and an identification cue")

    return TaskRequirements(
        status=Action.SEND,
        required=frozenset(required),
        reasons=tuple(reasons),
    )

