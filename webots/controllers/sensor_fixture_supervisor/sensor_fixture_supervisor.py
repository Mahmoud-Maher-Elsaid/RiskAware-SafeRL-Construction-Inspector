from controller import Supervisor

supervisor = Supervisor()
obstacle = supervisor.getFromDef("FIXTURE_OBSTACLE")
translation = obstacle.getField("translation")
timestep = int(supervisor.getBasicTimeStep())
distances = [0.80, 0.50, 0.30, 0.15, 10.0]
step = 0
while supervisor.step(timestep) != -1:
    phase = min(step // 20, len(distances) - 1)
    d = distances[phase]
    translation.setSFVec3f([0.0, 0.0, d + 0.025])
    step += 1
    if step >= 125:
        supervisor.simulationQuit(0)
        break
