"""BaseTrainer: bucle común de entrenamiento MBRL.

Recolecta episodios (con prefill aleatorio), entrena world model + actor-critic por
imaginación con `train_ratio` pasos de gradiente por episodio y gestiona checkpoint/resume y
logging. Las subclases (TeacherTrainer, StudentTrainer) solo construyen env/modelos y
completan los hooks de pérdida. El bucle no distingue profesor de alumno.
"""
import contextlib
import math
import os
import shutil
import time

import torch

from adcarla.policy.imagination import imagine_losses
from adcarla.training.agent import RandomAgent, collect_episode, flatten_states
from adcarla.training.replay_buffer import SequenceReplayBuffer
from adcarla.utils.clipping import adaptive_grad_clip_
from adcarla.utils.logging import MetricLogger


class BaseTrainer:
    """Bucle de entrenamiento compartido por el profesor y los dos alumnos.

    Cada subclase completa los hooks de construcción (`_build_env`, `_build_models`) y de
    pérdida (`compute_wm_loss`, `get_reward_fn`, `get_cont_fn`, y opcionalmente `collect_policy`
    y `get_teacher`).
    """

    def __init__(self, config: dict, env=None):
        """
        Args:
            config: configuración completa del experimento (ver configs/).
            env: entorno ya construido. Si es None se llama a
                `_build_env()`, que instancia CarlaEnv.
        """
        self.config = config
        self.train_config = config["train"]
        self.device = config.get("device", "cpu")
        self.name = config.get("name", "run")

        self.env = env if env is not None else self._build_env()
        self.num_actions = self.env.actions.n
        self.wm, self.actor, self.critic, self.agent = self._build_models()

        self.buffer = SequenceReplayBuffer(int(self.train_config["replay_capacity"]), int(self.train_config["seq_len"]))

        # Tasas de aprendizaje por componente.
        lr_wm = float(self.train_config.get("lr_wm", self.train_config["lr"]))
        lr_policy = float(self.train_config.get("lr_policy", self.train_config["lr"]))

        # Umbrales de recorte de gradientes
        self.grad_clip_wm = float(self.train_config["grad_clip_wm"])
        self.grad_clip_policy = float(self.train_config["grad_clip_policy"])

        # Modo de recorte:
        # - "norm" = clip_grad_norm_ con los dos umbrales de arriba
        # - "agc" = Adaptive Gradient Clipping de DreamerV3, cuyo umbral es relativo al peso.
        self.grad_clip_mode = str(self.train_config.get("grad_clip_mode", "norm")).lower()
        self.agc_clip_factor = float(self.train_config.get("agc_clip_factor", 0.3))
        self.agc_eps = float(self.train_config.get("agc_eps", 1.0e-3))

        # Optimizadores separados: actor_loss no puede actualizar critic.net (ni viceversa)
        self.opt_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr_wm, weight_decay=0.0)
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_policy, weight_decay=0.0)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_policy, weight_decay=0.0)

        self.checkpoint_dir = self.train_config.get("checkpoint_dir", "checkpoints/")
        total_ep = int(self.train_config.get("total_episodes", 0))
        self.logger = MetricLogger(self.config, name=self.name, total_episodes=total_ep)
        self.start_episode = 0   # lo cambia load_checkpoint() al reanudar

        # Configuración de grabación de vídeo durante el entrenamiento
        self._video_config = self.train_config.get("video", {})
        self._video_warned_no_rendering = False   # emitir el aviso solo una vez

    def _clip_gradients(self, module, threshold_norm: float) -> None:
        """Recorta los gradientes de `modulo` según el modo configurado: "norm" o "agc"."""
        if self.grad_clip_mode == "agc":
            adaptive_grad_clip_(module.parameters(), self.agc_clip_factor, self.agc_eps)
        else:
            torch.nn.utils.clip_grad_norm_(module.parameters(), threshold_norm)

    # ---- Hooks obligatorios (las subclases los implementan) ----
    def _build_env(self):
        raise NotImplementedError

    def _build_models(self):
        raise NotImplementedError

    def compute_wm_loss(self, batch: dict):
        raise NotImplementedError

    def get_reward_fn(self):
        raise NotImplementedError

    def get_cont_fn(self):
        raise NotImplementedError

    # ---- Hooks opcionales ----
    def collect_policy(self, episode: int) -> int:
        """Recoge un episodio con el agente propio del trainer.

        Args:
            episode: episodio actual, las subclases pueden usarlo para programar la recolección
                (el alumno decae `p_teacher` con él).

        Returns:
            Pasos actuados en el episodio (sin contar el step terminal), usado por train_ratio.
        """
        return collect_episode(self.env, self.agent, self.buffer, self.config, self.num_actions, greedy=False)

    def get_teacher(self):
        """World model que rueda en paralelo durante la imaginación (None = solo el propio)."""
        return None

    # ---- Checkpoint / resume ----
    def _checkpoint_state(self, episode: int) -> dict:
        return {
            "episode": episode,
            "run_dir": self.logger.run_dir,   # Para que --resume continúe la misma curva
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "config": self.config,
        }

    def save_checkpoint(self, episode: int, path: str = None):
        """Guarda pesos + optimizadores + episodio actual. `path` por defecto: `{checkpoint_dir}/{name}.pt`.

        El parámetro `episode` es el último episodio realmente completado, nunca el siguiente a
        ejecutar. Esta convención se coordina con `load_checkpoint` para mantener consistencia.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = path or os.path.join(self.checkpoint_dir, f"{self.name}.pt")
        torch.save(self._checkpoint_state(episode), path)

    def load_checkpoint(self, path: str = None):
        """Restaura pesos, optimizadores, el contador de episodios y el run_dir de logging (--resume).

        El parámetro `episode` en el checkpoint es el último completado (ver `save_checkpoint`),
        así que aquí se le suma 1 para obtener el siguiente pendiente.
        """
        path = path or os.path.join(self.checkpoint_dir, f"{self.name}.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt_wm.load_state_dict(ckpt["opt_wm"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.start_episode = int(ckpt["episode"]) + 1

        # Retoma el run_dir del checkpoint para que metrics.csv/train.log sigan siendo un
        # solo fichero con la curva completa, en vez de que MetricLogger (construido en __init__,
        # antes de saber si hay resume) abra un runs/<name>_<ts> nuevo y vacío en el episodio 400.
        run_dir = ckpt.get("run_dir")
        if run_dir and run_dir != self.logger.run_dir:
            self._switch_logger_run_dir(run_dir)

    def _switch_logger_run_dir(self, run_dir: str) -> None:
        """Cierra el logger recién creado en `__init__` y reabre uno sobre el `run_dir` del
        checkpoint, recargando las filas de su `metrics.csv`.

        Args:
            run_dir: carpeta objetivo leída del checkpoint. Se lee y se reabre.
        """
        new_run_dir = self.logger.run_dir
        self.logger.close()
        self.logger = MetricLogger(self.config, name=self.name, run_dir=run_dir,
                                    total_episodes=int(self.train_config.get("total_episodes", 0)))
        # El directorio con timestamp que se creó en __init__ no llegó a registrar ningún
        # episodio: se borra para no dejar una carpeta huérfana en runs/ en cada --resume.
        if new_run_dir != run_dir and not os.path.isfile(os.path.join(new_run_dir, "metrics.csv")):
            shutil.rmtree(new_run_dir, ignore_errors=True)

    # ---- Bucle principal ----
    def train(self, resume: bool = False):
        """Entrena hasta `train.total_episodes`. Guarda un checkpoint final al acabar (o al
        interrumpirse) y checkpoints periódicos cada `train.automatic_checkpoint_episodes`."""
        if resume:
            self.load_checkpoint()

        prefill_episodes = int(self.train_config.get("prefill_episodes", 0))
        train_ratio = float(self.train_config.get("train_ratio", 32))
        batch_size = int(self.train_config["batch_size"])
        seq_len = int(self.train_config["seq_len"])
        total_episodes = int(self.train_config["total_episodes"])
        automatic_checkpoint_episodes = int(self.train_config.get("automatic_checkpoint_episodes", 100))

        # 0 = desactivado. Por defecto cada 50 episodios se guarda un PNG de la máscara BEV (Ground-Truth vs Decoder)
        bev_vis_every = int(self.train_config.get("bev_vis_episodes", 50))

        # Ventana móvil del resumen: sin ella `summary()` promedia toda la ejecución y la curva
        # de aprendizaje sale plana.
        metrics_window = int(self.train_config.get("metrics_window", 50))
        random_agent = RandomAgent(self.num_actions)

        self.wm.train()
        self.actor.train()
        self.critic.train()

        # Rastrea el último episodio realmente completado. Si el bucle no llega a ejecutar
        # ninguno (p.ej. un --resume que ya estaba al final), no hay ninguno nuevo que registrar.
        self._last_completed_episode = self.start_episode - 1

        try:
            for episode in range(self.start_episode, total_episodes):
                if episode < prefill_episodes:
                    # Prefill: política aleatoria y sin ningún paso de gradiente hasta agotarlo.
                    collect_episode(self.env, random_agent, self.buffer, self.config, self.num_actions, greedy=False)
                    self._last_completed_episode = episode
                    continue

                episode_start_time = time.time()
                episode_length = self.collect_policy(episode)

                # Episodio de grabación: independiente del de entrenamiento, no va al buffer.
                # Se aisla para que no desplace el round-robin de rutas/tráfico ni contamine
                # env.metrics con la pasada greedy.
                with self._video_episode_isolation():
                    self._maybe_record_training_episode(episode)

                if not self.buffer.can_sample():
                    self._last_completed_episode = episode
                    continue

                # Número de actualizaciones de gradiente proporcional a los pasos de entorno recién
                # recogidos, no un único update por episodio.
                num_updates = math.ceil(episode_length * train_ratio / (batch_size * seq_len))
                # Se acumulan métricas de todas las actualizaciones y se promedian.
                accumulated = {}

                for _ in range(num_updates):
                    batch = self.buffer.sample(batch_size, self.device)

                    wm_loss, states, wm_metrics = self.compute_wm_loss(batch)
                    self.opt_wm.zero_grad()
                    wm_loss.backward()
                    self._clip_gradients(self.wm, self.grad_clip_wm)
                    self.opt_wm.step()

                    # flatten_states hace detach: los gradientes de la imaginación no se
                    # propagan de vuelta al paso de observación del world model.
                    start = flatten_states(states)
                    actor_loss, critic_loss, policy_metrics = imagine_losses(
                        self.wm.rssm, self.actor, self.critic, start,
                        self.get_reward_fn(), self.get_cont_fn(), self.config,
                        teacher=self.get_teacher())

                    self.opt_actor.zero_grad()
                    actor_loss.backward()
                    self._clip_gradients(self.actor, self.grad_clip_policy)
                    self.opt_actor.step()

                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    self._clip_gradients(self.critic, self.grad_clip_policy)
                    self.opt_critic.step()
                    self.critic.update_slow()   # EMA del target lento, tras el paso del optimizador

                    for key, value in {**wm_metrics, **policy_metrics}.items():
                        accumulated.setdefault(key, []).append(float(value))

                # Guardado de métricas
                means = {key: sum(v) / len(v) for key, v in accumulated.items()}
                last_episode_data = self.env.metrics.episodes[-1].as_dict() if self.env.metrics.episodes else {}
                self.logger.log(episode, {
                    **means,
                    **self.env.metrics.summary(window=metrics_window),
                    **{f"ep_{key}": value for key, value in last_episode_data.items()},
                    "num_updates": num_updates,
                    "episode_seconds": time.time() - episode_start_time,
                })

                # Guardado de un checkpoint del modelo
                if episode > 0 and episode % automatic_checkpoint_episodes == 0:
                    self.save_checkpoint(episode, os.path.join(self.checkpoint_dir, f"{self.name}_ep{episode}.pt"))

                # Snapshot BEV: guarda GT vs decoder para ver cómo aprende el world model a reconstruir
                if bev_vis_every > 0 and episode % bev_vis_every == 0 and "bev" in batch:
                    self._save_bev_snapshot(batch, states, episode)

                self._last_completed_episode = episode

        finally:
            self._finalize_training()

    def _save_bev_snapshot(self, batch: dict, states: dict, episode: int):
        """Guarda la máscara BEV generada del simulador y la generada por el decoder para
        el primer elemento del batch.

        Se llama desde el bucle principal cada `bev_vis_episodes` episodios con el
        último batch de la fase de actualización. Usa `torch.no_grad` para no interferir
        con el grafo de cómputo.

        El decoder devuelve logits crudos y `MetricLogger.save_bev` aplica sigmoid internamente.
        """
        from adcarla.carla_env.bev_privileged import CHANNELS as BEV_CHANNEL_NAMES
        try:
            with torch.no_grad():
                feat = self.wm.rssm.feat(states)            # [B, T, feat_dim]
                B, T = feat.shape[:2]
                feat_flat = feat.reshape(B * T, -1)
                pred_logits = self.wm.decoder(feat_flat)    # [B*T, C, H, W]
                pred_logits = pred_logits.reshape(
                    B, T, self.wm.bev_channels, self.wm.size, self.wm.size)
                
                # Primer elemento del batch, primer timestep
                self.logger.save_bev(
                    gt_bev=batch["bev"][0, 0],
                    pred_logits=pred_logits[0, 0],
                    episode=episode,
                    channel_names=BEV_CHANNEL_NAMES,
                )
        except Exception as exc:
            print(f"[{self.name}] advertencia: snapshot BEV ep {episode} fallido: {exc}")

    # ---- Grabación de vídeo durante el entrenamiento ----

    @contextlib.contextmanager
    def _video_episode_isolation(self):
        """Guarda y restaura el estado de `self.env` que el episodio de vídeo muta de paso.
        Aisla el episodio de grabación del flujo de entrenamiento principal.
        """
        env = self.env
        metrics = getattr(env, "metrics", None)
        n_episodes_before = len(metrics.episodes) if metrics is not None else None
        current_episode_before = getattr(metrics, "_current_episode", None) if metrics is not None else None
        route_episode_before = getattr(env, "_episode", None)
        traffic = getattr(env, "traffic", None)
        traffic_episode_before = getattr(traffic, "_episode", None) if traffic is not None else None
        traffic_seed_before = getattr(traffic, "episode_seed", None) if traffic is not None else None
        try:
            yield
        finally:
            if route_episode_before is not None:
                env._episode = route_episode_before
            if traffic is not None and traffic_episode_before is not None:
                traffic._episode = traffic_episode_before
                traffic.episode_seed = traffic_seed_before
            if metrics is not None and n_episodes_before is not None:
                del metrics.episodes[n_episodes_before:]
                metrics._current_episode = current_episode_before

    def _maybe_record_training_episode(self, episode: int) -> None:
        """Graba un episodio de vídeo si la configuración lo pide para este episodio.

        El episodio de grabación es extra. Se graba con política greedy, los frames
        van al ``VideoRecorder`` y los datos de observación se descartan (no van al buffer).
        Para el profesor (``carla.no_rendering: true``) se emite un aviso y se salta.

        Configuración (``train.video`` en el YAML):

            train:
              video:
                enabled: true
                record_every_n_episodes: 50
                fps: 10
                show_bev: false
                show_decoder: false
                weather: ClearNoon

        Si ``enabled`` es False o el episodio no toca, no hace nada.
        """
        config = self._video_config
        if not config.get("enabled", False):
            return
        every = int(config.get("record_every_n_episodes", 50))
        if every <= 0 or episode % every != 0:
            return

        # El profesor entrena con no_rendering=True: las cámaras no producen datos.
        if self.config.get("carla", {}).get("no_rendering", False):
            if not self._video_warned_no_rendering:
                print(
                    f"[{self.name}] aviso: grabación de vídeo desactivada porque "
                    "carla.no_rendering=true (el profesor entrena sin render). "
                    "Usar record_video.py tras el entrenamiento."
                )
                self._video_warned_no_rendering = True
            return

        try:
            self._record_training_video_episode(episode, config)
        except Exception as exc:
            print(f"[{self.name}] advertencia: episodio de vídeo ep {episode} fallido: {exc}")

    def _record_training_video_episode(self, episode: int, config: dict) -> None:
        """Rueda un episodio completo grabando cada frame al disco.
        """
        from adcarla.carla_env.bev_privileged import CHANNELS as BEV_CHANNEL_NAMES
        from adcarla.utils.video_recorder import VideoRecorder, _VideoCamera, VIDEO_CAM_H, VIDEO_CAM_W

        show_bev = bool(config.get("show_bev", False))
        show_decoder = bool(config.get("show_decoder", False))
        fps = int(config.get("fps", 10))
        weather = str(config.get("weather", "ClearNoon"))

        # Forzar BEV privilegiado en la observación si se va a mostrar
        if show_bev and not self.config.get("privileged_bev", False):
            self.config["privileged_bev"] = True
            # Registrar que se activó para esta grabación (no queremos que afecte al entrenamiento)
            _tmp_privileged_bev = True
        else:
            _tmp_privileged_bev = False

        # Ruta de salida dentro del directorio del run
        video_dir = os.path.join(self.logger.run_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        output_path = os.path.join(video_dir, f"ep{episode:05d}.mp4")

        obs = self.env.reset(weather=weather)
        self.agent.reset()

        # La cámara de vídeo se adjunta después de reset() porque env.ego cambia en cada reset.
        cam = _VideoCamera(self.env.world, self.env.ego, VIDEO_CAM_W, VIDEO_CAM_H)

        # Se pasa unos ticks para que la cámara reciba su primer frame.
        for _ in range(3):
            self.env.world.tick()

        with VideoRecorder(output_path, fps=fps, show_bev=show_bev, show_decoder=show_decoder) as rec:
            done = False
            step = 0
            reward = 0.0
            try:
                while not done:
                    action = self.agent.act(obs, greedy=True)
                    obs, reward, done, _ = self.env.step(action)
                    step += 1

                    rgb = cam.get_frame()
                    if rgb is None:
                        import numpy as np
                        rgb = np.zeros((VIDEO_CAM_H, VIDEO_CAM_W, 3), dtype=np.uint8)

                    bev_gt = None
                    if show_bev:
                        import numpy as np
                        raw = obs.get("bev_privileged")
                        if raw is not None:
                            import torch
                            bev_gt = torch.from_numpy(np.asarray(raw, dtype=np.float32))

                    decoder_logits = None
                    if show_decoder:
                        import torch
                        with torch.no_grad():
                            feat = self.wm.rssm.feat(self.agent.state)
                            decoder_logits = self.wm.decoder(feat)[0]

                    rec.add_frame(
                        rgb=rgb,
                        bev_gt=bev_gt,
                        decoder_logits=decoder_logits,
                        channel_names=BEV_CHANNEL_NAMES,
                        info={
                            "episode": episode,
                            "step": step,
                            "speed": getattr(self.env, "_last_speed", None),
                            "reward": float(reward),
                            "weather": weather,
                            "scenario": self.config.get("carla", {}).get("town", ""),
                        },
                    )
            finally:
                cam.destroy()

        # Deshacer la activación temporal de privileged_bev
        if _tmp_privileged_bev:
            self.config.pop("privileged_bev", None)

    def close(self):
        """Cierra el logger."""
        try:
            self.logger.close()
        except Exception:
            pass

    def _finalize_training(self):
        """Guarda checkpoint final, cierra el entorno y el logger. Llamado desde el `finally` de `train()`.

        Usa `self._last_completed_episode` (actualizado en cada iteración del bucle) para evitar
        guardar episodios que no se completaron realmente.
        """
        try:
            self.save_checkpoint(self._last_completed_episode)
        except Exception as error:
            print(f"[{self.name}] no se pudo guardar el checkpoint final: {error}")
        self.env.close()
        self.logger.close()
