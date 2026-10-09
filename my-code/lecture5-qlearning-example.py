import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Code for the Q-learning example of Lecture 5 (slide 18)

    import numpy as np

    Q = np.zeros((4, 2))

    R = np.zeros((4, 2))
    R[0,0] = 0
    R[0,1] = -1
    R[1,0] = -1
    R[1,1] = 1
    R[2,:] = 0
    R[3,:] = 0

    trajs = [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (1, 1), (3, 0)],
        [(0, 0), (1, 0), (2, 0)]
    ]

    cnt = 0
    traj_sas = []
    for traj in trajs:
        traj_sas.append([])
        for i in range(len(traj)-1):
            traj_sas[cnt].append(traj[i] + traj[i+1][:1])
        cnt+=1

    print(traj_sas)


    for i, ep in enumerate(traj_sas):
        for sas in ep:
            state = sas[0]
            action = sas[1]
            next_state = sas[2]
            Q[state, action] += R[state, action] + np.max(Q[next_state, :]) - Q[state, action]
        print(f"Iteration {i}")
        print(Q)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
