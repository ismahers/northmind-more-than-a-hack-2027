"""Reviewed fixed pilot, exclusively via scripts.run_local."""
def main():
    from possession.hsmm_pilot_run import run
    return run()

def verify_main():
    from possession.hsmm_pilot_run import verify, export
    verify()
    return export()

def replay_main():
    from possession.hsmm_pilot_run import replay
    return replay()
