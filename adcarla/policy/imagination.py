"""Entrenamiento por imaginación (estilo DreamerV3): rollout en el world model (sin CARLA) +
λ-returns + pérdidas actor/critic.

El actor y el critic nunca ven el simulador directamente: se entrenan sobre trayectorias
"imaginadas" por el RSSM (rssm.imagine, ver world_model/rssm.py) a partir de un start_state real.
reward_fn/cont_fn son las cabezas de predicción del world model (o del profesor vía Guidance
Mechanism) que estiman recompensa y probabilidad de continuar en cada paso imaginado.
"""
import torch
from adcarla.utils.distributions import two_hot_loss


def lambda_return(reward, value, cont, terminal_value, gamma=0.997, lambda_weight=0.95):
    """Calcula λ-returns (TD(λ)) bootstrapped, recorriendo la trayectoria hacia atrás.

    Cada return combina la recompensa inmediata con una mezcla entre el valor estimado del
    siguiente paso (bootstrap, peso 1-lambda_weight) y el propio return ya calculado del siguiente
    paso (peso lambda_weight) — lambda_weight alto favorece returns a más largo plazo (menos sesgo,
    más varianza).

    Args:
        reward: [B,H] recompensa estimada en cada paso imaginado (pasos 0..H-1).
        value: [B,H] valor estimado del critic en cada paso imaginado (pasos 0..H-1).
        cont: [B,H] probabilidad de continuar (1 - prob. de terminar el episodio) en cada paso;
            corta el bootstrap cuando el episodio termina.
        terminal_value: [B] V(H) — valor del critic un paso más allá del horizonte (bootstrap terminal).
        gamma: factor de descuento.
        lambda_weight: peso del bootstrap de TD(λ) (0 = solo un paso, 1 = Monte Carlo completo).

    Returns:
        [B,H] returns bootstrapped, uno por paso imaginado.
    """
    num_steps = reward.shape[1]
    running_return = terminal_value   # V(H): bootstrap del paso más allá del horizonte
    returns_list = []
    for t in reversed(range(num_steps)):
        # Valor de referencia del paso siguiente: V(t+1) si aún queda horizonte, V(H) en el último paso.
        bootstrap_value = value[:, t + 1] if t + 1 < num_steps else terminal_value
        running_return = reward[:, t] + gamma * cont[:, t] * ((1 - lambda_weight) * bootstrap_value + lambda_weight * running_return)
        returns_list.append(running_return)

    return torch.stack(list(reversed(returns_list)), dim=1)   # se revierte: se acumuló de t=H-1 a t=0


def imagine_losses(rssm, actor, critic, start_state, reward_fn, cont_fn, config):
    """Imagina `horizon` pasos desde start_state y devuelve (actor_loss, critic_loss, metrics).

    Args:
        rssm: RSSM ya entrenado (o durante el entrenamiento), provee imagine() y feat().
        actor: política discreta (Actor) usada para elegir la acción en cada paso imaginado.
        critic: Critic categórico (two-hot), estima el valor de cada paso imaginado.
        start_state: estado inicial real (dict h/stoch) desde el que arranca la imaginación.
        reward_fn: feat -> recompensa estimada [K]. Cabeza de recompensa del world model
            (o del profesor vía Guidance Mechanism).
        cont_fn: feat -> probabilidad de continuar [K] en [0,1]. Cabeza de "continue" del world model.
        config: config completa; se usa config["policy"] (horizon, gamma, lambda_, actor_entropy).

    Returns:
        actor_loss: escalar, pérdida de value gradient (DreamerV3) + bonus de entropía (a minimizar).
        critic_loss: escalar, two_hot_loss del critic contra los λ-returns (a minimizar).
        metrics: dict con valores medios para logging (return, value, losses, entropy).
    """
    policy_cfg = config["policy"]
    horizon = int(policy_cfg["horizon"])
    gamma = float(policy_cfg["gamma"])
    lambda_weight = float(policy_cfg["lambda_"])
    entropy_coef = float(policy_cfg["actor_entropy"])   # peso del bonus de entropía (fomenta exploración)

    # Rollout imaginado: el RSSM avanza en espacio latente con el actor, sin tocar CARLA.
    # actions no se usa con value gradient; solo se necesitan los estados y las entropías.
    states, _, entropies = rssm.imagine(actor, start_state, horizon)
    feat = rssm.feat(states)    # [B,H,feat_dim]

    with torch.no_grad():
        # reward, cont y terminal_value son señales fijas (targets): no se propaga gradiente.
        reward = reward_fn(feat)    # [B,H]
        cont = cont_fn(feat)        # [B,H] en [0,1]

        # V(H): bootstrap terminal un paso más allá del horizonte (estilo DreamerV3).
        # Se calcula avanzando un img_step desde el último estado imaginado.
        last_state = {"h": states["h"][:, -1], "stoch": states["stoch"][:, -1]}
        action_H, _ = actor(rssm.feat(last_state))
        state_H, _ = rssm.img_step(last_state, action_H)
        terminal_value = critic.value(rssm.feat(state_H))   # [B]
        
    value = critic.value(feat)      # [B,H] — con gradiente para el value gradient del actor
    returns = lambda_return(reward, value, cont, terminal_value, gamma, lambda_weight)

    # Actor: value gradient (DreamerV3), gradiente desde -returns → critic.value(feat) → feat
    # → rssm.imagine via straight-through → actor. Sin straight-through no habría gradiente.
    actor_loss = -returns.mean() - entropy_coef * entropies.mean()

    # Critic: se recalculan los logits desde feat.detach() para no propagar gradiente del critic al RSSM/actor.
    critic_logits = critic(feat.detach())
    critic_loss = two_hot_loss(critic_logits, returns.detach(), critic.bins).mean()

    metrics = {"return": returns.mean().item(), "value": value.mean().item(),
               "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
               "entropy": entropies.mean().item()}
    
    return actor_loss, critic_loss, metrics
