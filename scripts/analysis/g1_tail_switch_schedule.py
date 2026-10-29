"""Group assignments for two paired policy-switch interventions."""


def update_sign(group, time, switch_time):
    if group not in range(6) or time < 0 or switch_time < 1:
        raise ValueError('invalid switching schedule')
    if group >= 4 or (group in (2,3) and time >= switch_time):
        return 0
    return 1 if group % 2 == 0 else -1


def shared_prefix_noise(noise, base_count, replicates, time, switch_time, enabled):
    """Tie repeats within each base state only until the switch time.

    The caller draws the entire noise tensor before invoking this helper. The
    first dimension is organized in replicate blocks of ``base_count`` states.
    """
    if base_count < 1 or replicates < 1 or time < 0 or switch_time < 1:
        raise ValueError('invalid noise schedule')
    if len(noise) != base_count * replicates:
        raise ValueError('noise tensor does not match replicate schedule')
    if enabled and time < switch_time:
        return noise[:base_count].repeat(replicates, 1)
    return noise
