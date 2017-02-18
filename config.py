mean = (104, 117, 123)
scale = 300
steps = [s / scale for s in (8, 16, 32, 64, 100, 300)]
sizes = [s / scale for s in (30, 60, 111, 162, 213, 264, 315)]
aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
variance = (0.1, 0.2)
