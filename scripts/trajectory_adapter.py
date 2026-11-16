"""Only explicit public external paths; launched with stdout/stderr captured."""
from possession import trajectory_experiment as experiment


def run():
    return experiment.run()


def verify():
    return experiment.verify()


def replay():
    return experiment.replay()


def export():
    return experiment.export()
