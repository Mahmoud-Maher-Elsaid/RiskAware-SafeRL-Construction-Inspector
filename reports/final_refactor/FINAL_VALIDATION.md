# Final validation

The disposable Webots sensor fixture recorded direct physical responses of
0.8000, 0.5000, 0.3000, and 0.1500 m at the corresponding obstacle positions,
and 1.0000 m with no obstacle.  The production visible bounded run recorded
the same 1.0 m no-obstacle response and used the shared metre conversion.

The bounded visible runtime reached startup/controller/supervisor/worker
markers, rendered a sane 1185x762 client area, and the Supervisor camera trace
measured 0.70939 m robot translation against 0.70742 m Viewpoint translation.
The bounded recovery episode entered translational recovery and gained
0.20054 m.

The 1,000-decision integration smoke completed with zero collisions, zero stale
observations, zero stale recurrent state, 0.999 valid-target fraction, 8
visited cells, 3.0593 m displacement, and 1.35 maximum commanded wheel speed.

The controlled obstacle-approach fixture now runs through the full
hierarchical controller. Three deterministic runs began at 0.44997 m,
reached 0.01706 m, recorded 46 actuator-backstop interventions each, and had
zero collisions. At minimum clearance the final command was [0, 0].

A current managed 1,000-decision smoke completed with 3.1029 m displacement,
3.2973 m path length, 9 visited cells, 0.999 valid-target fraction, zero
collisions, zero stale observations, and a 1.35 command maximum.

The final post-refactor 10,000-decision managed run completed from the current
runtime. It recorded 5.4691 m displacement, 17.1745 m path length, 34 visited
cells, 0.9997 valid-target fraction, target-none maximum streak 1, 8,415
replans, stuck-event fraction 0.0191, four spin events with maximum duration
10 decisions, three recovery episodes (two successful), zero collisions, zero
stale observations, zero stale policy state, and a 1.35 maximum wheel command.

The emergency actuator threshold is 0.40 m. With a 0.62 m body (0.31 m
forward extent), 32 ms control step, 1.35 maximum command (0.0432 m per
step), and a 0.05 m dynamic margin, this is the smallest round threshold that
covers body extent plus one-step travel and margin. Three post-threshold
controlled safety runs recorded 49 backstop interventions each, a 0.01476 m
sensor minimum caused by the validation obstacle repositioning, and zero
collisions.

The final post-counter-fix bounded 60-second user-runtime run completed with 1.8896 m
displacement, 1.9076 m path length, 5 visited cells, zero collisions, zero
spin events, stuck fraction 0.0117, maximum command 1.35, zero stale state,
both turn directions, and recovery success. Its physical camera trace measured
1.8896 m robot translation versus 1.8760 m Viewpoint translation, with camera
rotation/level/forward gates passing.
