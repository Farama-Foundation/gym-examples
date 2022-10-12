"""Define time measurement functions."""


def tic():
    """Start the global timer."""
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    """End the global timer and print."""
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() -
              startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
