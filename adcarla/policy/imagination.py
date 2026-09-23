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
    paso (peso lambda_weight). Lambda_weight alto favorece returns a más largo plazo (menos sesgo,
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


def imagine_losses(rssm, actor, critic, start_state, reward_fn, cont_fn, config, teacher=None):
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
        teacher: opcional. Segundo world model que rueda en paralelo con la misma acción y la
            misma muestra estocástica; `reward_fn`/`cont_fn` se evalúan entonces sobre su feat
            (pseudo-deducción, Raw2Drive Fig. 5) y su `distill_loss` se suma a la pérdida del
            actor.

    Returns:
        actor_loss: escalar, estimador REINFORCE con baseline del critic + bonus de entropía
            (a minimizar).
        critic_loss: escalar, two_hot_loss del critic contra los λ-returns más el regularizador
            hacia su target lento (a minimizar).
        metrics: dict con valores medios para logging (return, value, advantage, return_scale,
            losses, entropy, y "distill" si hay teacher).
    """
    policy_cfg = config["policy"]
    horizon = int(policy_cfg["horizon"])
    gamma = float(policy_cfg["gamma"])
    lambda_weight = float(policy_cfg["lambda_"])
    entropy_coef = float(policy_cfg["actor_entropy"])   # peso del bonus de entropía (fomenta exploración)

    # Rollout imaginado: el RSSM avanza en espacio latente con el actor, sin tocar CARLA. Los
    # estados y las acciones salen sin gradiente (ver RSSM.imagine), log_probs y entropies no.
    states, _, log_probs, entropies = rssm.imagine(actor, start_state, horizon,
                                                   on_step=None if teacher is None else teacher.step)
    feat = rssm.feat(states)    # [B,H,feat_dim]

    # Pseudo-deducción: con teacher las cabezas se evalúan sobre el feat del profesor, no sobre
    # el del alumno (que aún no vive en el mismo espacio latente).
    head_feat = feat if teacher is None else teacher.feats()

    # Todo lo que entra en la ventaja es un target fijo: con REINFORCE el actor no deriva a través
    # de reward/cont/value.
    with torch.no_grad():
        reward = reward_fn(head_feat)    # [B,H]
        cont = cont_fn(head_feat)        # [B,H] en [0,1]

        # V(H): bootstrap terminal un paso más allá del horizonte (estilo DreamerV3).
        # Se calcula avanzando un img_step desde el último estado imaginado.
        last_state = {"h": states["h"][:, -1], "stoch": states["stoch"][:, -1]}
        action_H, _, _ = actor(rssm.feat(last_state))
        state_H, _ = rssm.img_step(last_state, action_H)
        terminal_value = critic.value(rssm.feat(state_H))   # [B]

        value = critic.value(feat)      # [B,H] — baseline de la ventaja
        returns = lambda_return(reward, value, cont, terminal_value, gamma, lambda_weight)

        # Ec. 7: la ventaja se divide por max(1, S) con S = EMA(perc95 - perc5). Sin esto el
        # coeficiente de entropía (3e-4) no tiene escala de referencia.
        advantage = (returns - value) / actor.return_norm.update(returns)

    # Actor: estimador REINFORCE con baseline del critic (DreamerV3 Ec. 6). El gradiente entra
    # solo por log π(a|s) y por la entropía, la ventaja es un escalar ya fijado.
    actor_loss = -(advantage * log_probs).mean() - entropy_coef * entropies.mean()

    distill_loss = None
    if teacher is not None:
        # Destilación: CE contra la distribución de acciones del profesor en el mismo paso.
        # `feat.detach()`: la CE solo debe mover los pesos del actor. Sin el detach el gradiente
        # tendría un segundo camino, empujando al actor a llevar el rollout hacia estados donde el
        # objetivo (ya fijado) sea fácil, y dejaría de ser behaviour cloning.
        distill_loss = teacher.distill_loss(actor.logits(feat.detach()))
        actor_loss = actor_loss + distill_loss

    # Critic: `feat.detach()` para no propagar gradiente al RSSM.
    critic_logits = critic(feat.detach())
    critic_loss = two_hot_loss(critic_logits, returns, critic.bins).mean()
    # Regularizador hacia el target lento (DreamerV3, "Critic learning"): el critic regresa targets
    # que dependen de sus propias predicciones, así que sin esto el bootstrap se persigue a sí mismo.
    slow_loss = two_hot_loss(critic_logits, critic.slow_value(feat), critic.bins).mean()
    critic_loss = critic_loss + slow_loss

    metrics = {"return": returns.mean().item(), "value": value.mean().item(),
               "advantage": advantage.mean().item(),
               "return_scale": float(actor.return_norm.scale),
               "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
               "critic_slow": slow_loss.item(), "entropy": entropies.mean().item()}
    if distill_loss is not None:
        metrics["distill"] = distill_loss.item()

    return actor_loss, critic_loss, metrics
