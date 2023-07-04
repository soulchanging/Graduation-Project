import numpy as np
from env.core import World, Agent, Landmark
from env.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.num_agents = args.num_agents  # 1
        assert world.num_agents == 1, (
            "only 1 agents is supported, check the config.py.")
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.1
            agent.max_speed = 1
        # add landmarks
        world.num_landmarks = args.num_landmarks  # 1
        assert world.num_landmarks == 1, (
            "only 1 landmarks is supported, check the config.py.")
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            # agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_pos = np.array([0.0,0.0])
            agent.init_p_pos = agent.state.p_pos.copy()
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[0].state.p_pos = np.array([0.9, 0.9])
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        dists = np.zeros((world.num_agents, world.num_landmarks))
        for ia, a in enumerate(world.agents):
            for il, l in enumerate(world.landmarks):
                dists[ia][il] = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                init_dist = np.sqrt(np.sum(np.square(a.init_p_pos - l.state.p_pos)))
                if init_dist > 0.1:
                    dists[ia][il] /= init_dist
        # rew = np.sum(np.exp(-np.min(dists, axis=1) * 10))
        rew = -np.sum(np.min(dists, axis=1))

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + entity_color + other_pos)

    def info(self, agent, world):
        dists = []
        for l in world.landmarks:
            dists.append(np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))))
        closest_id = np.argmin(dists)
        return {'target': closest_id}
