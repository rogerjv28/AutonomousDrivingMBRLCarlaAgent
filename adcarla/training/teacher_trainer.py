"""Etapa 1: entrenamiento del World Model privilegiado y la política del profesor. Requiere CARLA.

Por iteración: (1) recoger episodio en CARLA, (2) entrenar world model (recon+reward+cont+KL),
(3) entrenar actor-critic por imaginación. El checkpoint resultante guía al alumno en la Etapa 2.
"""
import torch

from adcarla.carla_env.env import CarlaEnv
from adcarla.encoders.factory import build_encoder
from adcarla.world_model.world_model import WorldModel
from adcarla.policy.actor_critic import Actor, Critic
from adcarla.policy.imagination import imagine_losses
from adcarla.training.replay_buffer import SequenceReplayBuffer
from adcarla.training.agent import RolloutAgent, obs_to_step, flatten_states
from adcarla.utils.distributions import from_probs, symexp


def train_teacher(config: dict):
    """Entrena el World Model privilegiado y la política del profesor en CARLA.

    Args:
        config: configuración completa (ver configs/teacher.yaml).
            Se añade automáticamente "encoder": "privileged" y "privileged_bev": True.

    Returns:
        Tupla (wm, actor, critic) con los módulos entrenados. Guarda checkpoints/teacher.pt.
    """
    device = config.get("device", "cpu")
    config = {**config, "encoder": "privileged", "privileged_bev": True}
    train_config = config["train"]

    # --- Inicialización ---
    env = CarlaEnv(config)
    num_actions = env.actions.n
    teacher_wm = WorldModel(config, build_encoder(config), num_actions).to(device)
    actor = Actor(teacher_wm.rssm.feat_dim, num_actions).to(device)
    critic = Critic(teacher_wm.rssm.feat_dim, int(config["world_model"]["num_bins"])).to(device)
    replay_buffer = SequenceReplayBuffer(int(train_config["replay_capacity"]), int(train_config["seq_len"]))
    agent = RolloutAgent(teacher_wm, actor, config, device)

    # Optimizadores separados: actor_loss no puede actualizar critic.net (ni viceversa)
    opt_wm    = torch.optim.Adam(teacher_wm.parameters(), lr=float(train_config["lr"]))
    opt_actor = torch.optim.Adam(actor.parameters(), lr=float(train_config["lr"]))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=float(train_config["lr"]))

    # Cabezas del World Model del profesor como funciones de reward/cont para la imaginación.
    def reward_fn(feat):
        return symexp(from_probs(torch.softmax(teacher_wm.reward(feat), -1), teacher_wm.bins))

    def cont_fn(feat):
        return torch.sigmoid(teacher_wm.cont(feat))

    # Ponemos los tres módulos del agente profesor en modo train antes de entrenar.
    teacher_wm.train()
    actor.train()
    critic.train()

    # --- Bucle de entrenamiento ---
    episode = 0    # inicialización
    try:
        for episode in range(int(train_config["total_episodes"])):

            # Inicializar episodio con exploración (greedy=False → muestreo categórico)
            observation = env.reset()
            agent.reset()
            done = False

            # Ejecución episodio completo
            while not done:
                action = agent.act(observation, greedy=False)
                next_obs, reward, done, _ = env.step(action)
                step_data = obs_to_step(observation, config)
                step_data.update(action=action, reward=reward, cont=0.0 if done else 1.0)
                replay_buffer.add_step(step_data)
                observation = next_obs

            # Guarda el episodio si cumple las condiciones en end_episode()
            replay_buffer.end_episode()

            if not replay_buffer.can_sample():
                continue

            # --- World Model ---
            batch = replay_buffer.sample(int(train_config["batch_size"]), device)
            wm_loss, states, wm_metrics = teacher_wm.loss(batch)    # states: [B, T, dim] con gradiente

            # Borra gradientes antiguos, propaga la perdida hacia atras y actualiza los pesos
            opt_wm.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_wm.parameters(), max_norm=1000.0)  # DreamerV3: previene gradientes muy altos en secuencias largas
            opt_wm.step()

            # --- Actor-Critic por imaginación ---
            # flatten_states hace detach de los estados: los gradientes de la imaginación no se propagan
            # de vuelta al paso de observación del World Model
            start = flatten_states(states)  # [B*T, dim]
            actor_loss, critic_loss, policy_metrics = imagine_losses(teacher_wm.rssm, actor, critic, start, reward_fn, cont_fn, config)

            # Actor
            # Borra gradientes antiguos, propaga la perdida hacia atras y actualiza los pesos
            opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1000.0)
            opt_actor.step()

            # Critic
            # Borra gradientes antiguos, propaga la perdida hacia atras y actualiza los pesos
            opt_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1000.0)
            opt_critic.step()

            if episode % 50 == 0:
                print(
                    f"[teacher ep={episode}] "
                    f"wm={wm_metrics['loss']:.3f} recon={wm_metrics['recon']:.3f} "
                    f"reward={wm_metrics['reward']:.3f} cont={wm_metrics['cont']:.3f} "
                    f"kl={wm_metrics['kl']:.3f} | "
                    f"actor={policy_metrics['actor_loss']:.3f} critic={policy_metrics['critic_loss']:.3f} "
                    f"entropy={policy_metrics['entropy']:.3f} return={policy_metrics['return']:.2f}"
                )

            # Checkpoint periódico: permite reanudar el entrenamiento si se interrumpe
            automatic_checkpoint_episodes = int(train_config.get("automatic_checkpoint_episodes", 500))
            if episode > 0 and episode % automatic_checkpoint_episodes == 0:
                torch.save({
                    "episode": episode,
                    "wm": teacher_wm.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "opt_wm": opt_wm.state_dict(),
                    "opt_actor": opt_actor.state_dict(),
                    "opt_critic": opt_critic.state_dict(),
                    "config": config,
                }, f"checkpoints/teacher_ep{episode}.pt")

        # Guarda pesos, optimizadores, episodio actual y config para poder reanudar el entrenamiento
        torch.save({
            "episode": episode,
            "wm": teacher_wm.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "opt_wm": opt_wm.state_dict(),
            "opt_actor": opt_actor.state_dict(),
            "opt_critic": opt_critic.state_dict(),
            "config": config,
        }, "checkpoints/teacher.pt")

    finally:
        env.close()
    return teacher_wm, actor, critic
