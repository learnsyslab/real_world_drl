"""Eval-only actor that mirrors SACActor.run/run_pe but records per-episode
forces (N), torques (Nm), and cycle time (s), and prompts the user for
ground-truth success at the end of each episode.

Results are written incrementally to a JSON file so partial runs survive
Ctrl+C. The full schema is documented at the bottom of this file.
"""

import json
import logging
import os
import time

import numpy as np
import torch.multiprocessing as mp

from crisp_drl.agents.shared.actor import SACActor
from crisp_drl.data import utils


FT_KEY = "observation.state.sensors_bota_ft_sensor"
CARTESIAN_KEY = "observation.state.cartesian"


def _atomic_write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _wrap_to_pi(a: float) -> float:
    """Wrap angle (rad) to [-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        "max": float(arr.max()),
    }


class EvalSACActor(SACActor):
    """SAC actor subclass with metrics + human-in-the-loop success prompt."""

    def __init__(
        self,
        args,
        config,
        parameters_queue: mp.Queue,
        run_name: str,
        env,
        rew_fn,
        eval_json_path: str,
        prompt_user: bool = True,
    ):
        super().__init__(args, config, parameters_queue, run_name, env, rew_fn)
        self.eval_json_path = eval_json_path
        self.prompt_user = prompt_user
        self.episodes_data: list[dict] = []
        self.n_unusable_skipped = 0
        self._started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        out_dir = os.path.dirname(self.eval_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        logging.info(f"[eval] results will be written to: {self.eval_json_path}")

        # GT references for pose-diff diagnostics.
        try:
            self.goal_gt_xyz = np.asarray(
                self.config.goal_position_ground_truth, dtype=np.float64
            ).reshape(-1)[:3]
        except (AttributeError, TypeError, ValueError):
            self.goal_gt_xyz = None
        try:
            self.goal_gt_rpy = np.asarray(
                self.config.goal_orientation_ground_truth_euler, dtype=np.float64
            ).reshape(-1)[:3]
        except (AttributeError, TypeError, ValueError):
            self.goal_gt_rpy = None

    @staticmethod
    def _ft_from_obs(obs_dict) -> "np.ndarray | None":
        if not isinstance(obs_dict, dict):
            return None
        ft = obs_dict.get(FT_KEY)
        if ft is None:
            return None
        arr = np.asarray(ft, dtype=np.float64).reshape(-1)
        if arr.size < 6:
            return None
        return arr[:6].copy()

    @staticmethod
    def _cartesian_from_obs(obs_dict) -> "np.ndarray | None":
        """Raw 6-vec [x,y,z,roll,pitch,yaw] from the unformatted obs dict."""
        if not isinstance(obs_dict, dict):
            return None
        cart = obs_dict.get(CARTESIAN_KEY)
        if cart is None:
            return None
        arr = np.asarray(cart, dtype=np.float64).reshape(-1)
        if arr.size < 6:
            return None
        return arr[:6].copy()

    def _pose_diff(self, start_cart, reset_info):
        """Build a dict comparing RL-start EE pose to GT goal and PE goal.

        - start_cart  : 6-vec [x,y,z,roll,pitch,yaw], captured right after
                        env.reset() returns (i.e. when RL policy is about to
                        take its first step).
        - reset_info  : dict from env.reset; for PE wrapper has
                        'reset.goal_position' (3-vec) and
                        'reset.goal_orientation.rotation_z' (float).
        """
        pd: dict = {}
        if start_cart is None:
            pd["start_xyz_m"] = None
            pd["start_rpy_rad"] = None
        else:
            pd["start_xyz_m"] = start_cart[:3].tolist()
            pd["start_rpy_rad"] = start_cart[3:6].tolist()

        if self.goal_gt_xyz is not None:
            pd["goal_gt_xyz_m"] = self.goal_gt_xyz.tolist()
        if self.goal_gt_rpy is not None:
            pd["goal_gt_rpy_rad"] = self.goal_gt_rpy.tolist()

        est_xyz = None
        est_rz = None
        if isinstance(reset_info, dict):
            v = reset_info.get("reset.goal_position")
            if v is not None:
                try:
                    est_xyz = np.asarray(v, dtype=np.float64).reshape(-1)[:3]
                except Exception:
                    est_xyz = None
            v = reset_info.get("reset.goal_orientation.rotation_z")
            if v is not None:
                try:
                    est_rz = float(v)
                except Exception:
                    est_rz = None

        pd["goal_estimated_xyz_m"] = est_xyz.tolist() if est_xyz is not None else None
        pd["goal_estimated_rz_rad"] = est_rz

        # --- distances (mm / deg) -------------------------------------- #
        def _dist_mm(a, b, idx):
            if a is None or b is None:
                return None
            try:
                return float(np.linalg.norm(np.asarray(a)[idx] - np.asarray(b)[idx]) * 1000.0)
            except Exception:
                return None

        s_xy = start_cart[:2] if start_cart is not None else None
        s_xyz = start_cart[:3] if start_cart is not None else None
        gt_xy = self.goal_gt_xyz[:2] if self.goal_gt_xyz is not None else None
        gt_xyz = self.goal_gt_xyz if self.goal_gt_xyz is not None else None
        est_xy = est_xyz[:2] if est_xyz is not None else None

        pd["start_to_gt_goal_xy_mm"] = _dist_mm(s_xy, gt_xy, slice(None))
        pd["start_to_gt_goal_xyz_mm"] = _dist_mm(s_xyz, gt_xyz, slice(None))
        pd["start_to_gt_goal_z_mm"] = (
            float((start_cart[2] - self.goal_gt_xyz[2]) * 1000.0)
            if (start_cart is not None and self.goal_gt_xyz is not None)
            else None
        )
        pd["start_to_est_goal_xy_mm"] = _dist_mm(s_xy, est_xy, slice(None))
        pd["est_to_gt_goal_xy_mm"] = _dist_mm(est_xy, gt_xy, slice(None))

        # --- yaw diagnostics (deg) ------------------------------------- #
        start_yaw = float(start_cart[5]) if start_cart is not None else None
        gt_yaw = (
            float(self.goal_gt_rpy[2])
            if (self.goal_gt_rpy is not None and self.goal_gt_rpy.size >= 3)
            else None
        )
        pd["start_yaw_deg"] = (
            float(np.rad2deg(start_yaw)) if start_yaw is not None else None
        )
        pd["est_goal_yaw_deg"] = (
            float(np.rad2deg(est_rz)) if est_rz is not None else None
        )
        pd["gt_goal_yaw_deg"] = float(np.rad2deg(gt_yaw)) if gt_yaw is not None else None
        pd["start_minus_est_yaw_deg"] = (
            float(np.rad2deg(_wrap_to_pi(start_yaw - est_rz)))
            if (start_yaw is not None and est_rz is not None)
            else None
        )
        pd["start_minus_gt_yaw_deg"] = (
            float(np.rad2deg(_wrap_to_pi(start_yaw - gt_yaw)))
            if (start_yaw is not None and gt_yaw is not None)
            else None
        )
        return pd

    def _ask_user_success(self, episode_idx: int):
        if not self.prompt_user:
            return None
        while True:
            try:
                ans = input(
                    f"\n[eval] Episode {episode_idx} — success? [s=success / n=not]: "
                ).strip().lower()
            except EOFError:
                logging.warning("[eval] stdin closed; recording user_success=None")
                return None
            if ans == "s":
                return True
            if ans == "n":
                return False
            print("  please type 's' or 'n'.")

    @staticmethod
    def _ft_stats(ft_log):
        if not ft_log:
            return (
                {"mean": None, "max": None},
                {"mean": None, "max": None},
            )
        arr = np.asarray(ft_log, dtype=np.float64)
        f_xyz, t_xyz = arr[:, 0:3], arr[:, 3:6]
        force_norm = np.linalg.norm(f_xyz, axis=1)
        torque_norm = np.linalg.norm(t_xyz, axis=1)
        force = {
            "mean": float(force_norm.mean()),
            "max": float(force_norm.max()),
            "fx_mean": float(f_xyz[:, 0].mean()),
            "fy_mean": float(f_xyz[:, 1].mean()),
            "fz_mean": float(f_xyz[:, 2].mean()),
            "fx_max_abs": float(np.max(np.abs(f_xyz[:, 0]))),
            "fy_max_abs": float(np.max(np.abs(f_xyz[:, 1]))),
            "fz_max_abs": float(np.max(np.abs(f_xyz[:, 2]))),
        }
        torque = {
            "mean": float(torque_norm.mean()),
            "max": float(torque_norm.max()),
            "tx_mean": float(t_xyz[:, 0].mean()),
            "ty_mean": float(t_xyz[:, 1].mean()),
            "tz_mean": float(t_xyz[:, 2].mean()),
            "tx_max_abs": float(np.max(np.abs(t_xyz[:, 0]))),
            "ty_max_abs": float(np.max(np.abs(t_xyz[:, 1]))),
            "tz_max_abs": float(np.max(np.abs(t_xyz[:, 2]))),
        }
        return force, torque

    def _make_episode_record(
        self,
        ft_log,
        cycle_time_s,
        length,
        classifier_succ,
        user_succ,
        reset_info,
        start_cart,
    ) -> dict:
        force, torque = self._ft_stats(ft_log)
        rec = {
            "episode": len(self.episodes_data),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "length_steps": int(length),
            "cycle_time_s": float(cycle_time_s),
            "user_success": user_succ,
            "classifier_success": bool(classifier_succ),
            "force_N": force,
            "torque_Nm": torque,
            "pose_diff": self._pose_diff(start_cart, reset_info),
        }
        if isinstance(reset_info, dict):
            for key in (
                "reset.grasped.delta",
                "reset.grasped.delta_estimated",
                "reset.goal_position.offset",
            ):
                v = reset_info.get(key)
                if v is None:
                    continue
                try:
                    rec[key] = np.asarray(v).tolist()
                except Exception:
                    rec[key] = v
        return rec

    def _compute_summary(self, eps_list=None, include_unusable: bool = True) -> dict:
        """Summarize a list of episode records.

        Args:
            eps_list: explicit list of records (e.g. a 10-episode chunk).
                      Defaults to all completed episodes.
            include_unusable: whether to attach the global unusable-rollout
                              count. Set False for chunk summaries since it
                              applies to the whole run, not the chunk.
        """
        eps = self.episodes_data if eps_list is None else eps_list
        n = len(eps)
        summary: dict = {"n_episodes": n}
        if include_unusable:
            summary["n_unusable_skipped"] = self.n_unusable_skipped
        if n == 0:
            return summary

        user_arr = [e["user_success"] for e in eps if e["user_success"] is not None]
        clf_arr = [bool(e["classifier_success"]) for e in eps]
        agree = [
            int(bool(e["user_success"]) == bool(e["classifier_success"]))
            for e in eps
            if e["user_success"] is not None
        ]

        force_mean = np.array(
            [e["force_N"]["mean"] for e in eps if e["force_N"]["mean"] is not None]
        )
        force_max = np.array(
            [e["force_N"]["max"] for e in eps if e["force_N"]["max"] is not None]
        )
        torque_mean = np.array(
            [e["torque_Nm"]["mean"] for e in eps if e["torque_Nm"]["mean"] is not None]
        )
        torque_max = np.array(
            [e["torque_Nm"]["max"] for e in eps if e["torque_Nm"]["max"] is not None]
        )
        cycle = np.array([e["cycle_time_s"] for e in eps], dtype=np.float64)

        def stats(arr):
            if arr.size == 0:
                return {"mean": None, "std": None, "max": None}
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
                "max": float(arr.max()),
            }

        summary["success_rate_user"] = (
            (sum(user_arr) / len(user_arr)) if user_arr else None
        )
        summary["n_user_labelled"] = len(user_arr)
        summary["success_rate_classifier"] = sum(clf_arr) / n
        summary["agreement_rate"] = (sum(agree) / len(agree)) if agree else None
        summary["force_N"] = {
            "episode_mean_stats": stats(force_mean),
            "episode_max_stats": stats(force_max),
        }
        summary["torque_Nm"] = {
            "episode_mean_stats": stats(torque_mean),
            "episode_max_stats": stats(torque_max),
        }
        summary["cycle_time_s"] = stats(cycle)

        # --- pose-diff aggregates, split by outcome --------------------- #
        pose_metric_keys = [
            "start_to_gt_goal_xy_mm",
            "start_to_gt_goal_xyz_mm",
            "start_to_gt_goal_z_mm",
            "start_to_est_goal_xy_mm",
            "est_to_gt_goal_xy_mm",
            "start_minus_gt_yaw_deg",
            "start_minus_est_yaw_deg",
        ]

        def _bucket(filter_fn):
            out: dict = {"n": 0}
            sel = [e for e in eps if filter_fn(e)]
            out["n"] = len(sel)
            for k in pose_metric_keys:
                vals = [
                    e["pose_diff"][k]
                    for e in sel
                    if "pose_diff" in e and e["pose_diff"].get(k) is not None
                ]
                arr = np.asarray(vals, dtype=np.float64)
                out[k] = _stats(arr) if arr.size else {"n": 0, "mean": None, "std": None, "max": None}
            return out

        summary["pose_diff_overall"] = _bucket(lambda e: True)
        summary["pose_diff_by_classifier"] = {
            "success": _bucket(lambda e: bool(e["classifier_success"])),
            "failure": _bucket(lambda e: not bool(e["classifier_success"])),
        }
        if any(e["user_success"] is not None for e in eps):
            summary["pose_diff_by_user"] = {
                "success": _bucket(lambda e: e["user_success"] is True),
                "failure": _bucket(lambda e: e["user_success"] is False),
            }
        return summary

    def _chunked_summaries(self, chunk_size: int = 10) -> list:
        """One summary per ``chunk_size``-episode block, in episode order."""
        chunks = []
        for i in range(0, len(self.episodes_data), chunk_size):
            block = self.episodes_data[i : i + chunk_size]
            if not block:
                continue
            chunks.append(
                {
                    "chunk_index": i // chunk_size,
                    "episode_range": [block[0]["episode"], block[-1]["episode"]],
                    "summary": self._compute_summary(
                        eps_list=block, include_unusable=False
                    ),
                }
            )
        return chunks

    def _flush(self) -> None:
        payload = {
            "meta": {
                "load_policy": getattr(self.args, "load_policy", None),
                "task": getattr(self.args, "task", None),
                "success_threshold": getattr(self.args, "success_threshold", None),
                "snap_reinforce": getattr(self.args, "snap_reinforce", False),
                "use_pose_estimation": getattr(
                    self.args, "use_pose_estimation", False
                ),
                "pe_3dof": getattr(self.args, "pe_3dof", False),
                "max_episodes": getattr(self.args, "max_episodes", None),
                "prompt_user": self.prompt_user,
                "n_cameras": self.config.n_cameras,
                "actor_output_dim": self.config.actor_output_dim,
                "actor_nonvision_input_dim": self.config.actor_nonvision_input_dim,
                "max_action": np.asarray(self.config.max_action).tolist(),
                "started_at": self._started_at,
                "now": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            },
            "episodes": self.episodes_data,
            "summary_per_10_episodes": self._chunked_summaries(chunk_size=10),
            "summary": self._compute_summary(),
        }
        _atomic_write_json(self.eval_json_path, payload)

    def run_eval(self, data_queue: mp.Queue) -> None:
        """Eval loop: mirrors SACActor.run / run_pe, adds FT + cycle-time +
        human-success logging. Counts only valid episodes (E_ROLLOUT_UNUSABLE
        rollouts are skipped and retried, matching SACActor.run behaviour)."""
        max_eps = max(int(getattr(self.args, "max_episodes", 0) or 0), 0)

        try:
            obs_dict, reset_info = self.env.reset(seed=self.config.seed)
            start_cart = self._cartesian_from_obs(obs_dict)
            obs = obs_dict["observation.formatted"]
            actual_grasp_pos = reset_info["reset.grasped.position"]
            current_reset_info = reset_info
            current_start_cart = start_cart

            all_actions: list = []
            all_rewards: list = []
            all_observations: list = [obs]
            all_infos: list = [reset_info]

            ft_log: list = []
            episode_length = 0
            t_episode_start = time.perf_counter()
            self.episode_num = 0

            while True:
                self.global_step += 1
                obs_input = utils.shared_encode(
                    self.shared_encoder,
                    obs.view(1, -1),
                    self.config.shared_encoder_gradient,
                )
                action, _ = self.actor(obs_input)
                action = action.view(-1).detach()
                all_actions.append(action)

                obs_dict, reward, termination, truncation, info = self.env.step(
                    action.cpu().numpy()
                )

                ft = self._ft_from_obs(obs_dict)
                if ft is not None:
                    ft_log.append(ft)

                obs = obs_dict["observation.formatted"]
                all_observations.append(obs)
                all_infos.append(info)
                all_rewards.append(reward)

                episode_length += 1
                done = termination or truncation
                if not done:
                    continue

                event_names = []
                if isinstance(info, dict) and "custom_events" in info:
                    event_names = [e[1] for e in info["custom_events"]]
                if "E_ROLLOUT_UNUSABLE" in event_names:
                    self.n_unusable_skipped += 1
                    logging.warning(
                        f"[eval] rollout unusable; retrying "
                        f"(skipped={self.n_unusable_skipped})"
                    )
                    self.global_step -= episode_length
                    obs_dict, reset_info = self.env.reset()
                    start_cart = self._cartesian_from_obs(obs_dict)
                    obs = obs_dict["observation.formatted"]
                    actual_grasp_pos = reset_info["reset.grasped.position"]
                    current_reset_info = reset_info
                    current_start_cart = start_cart
                    all_actions = []
                    all_rewards = []
                    all_observations = [obs]
                    all_infos = [reset_info]
                    ft_log = []
                    episode_length = 0
                    t_episode_start = time.perf_counter()
                    continue

                cycle_time_s = time.perf_counter() - t_episode_start

                all_actions, all_observations, all_rewards, all_infos = self.reward_fn(
                    all_actions,
                    all_observations,
                    all_rewards,
                    all_infos,
                    actual_grasp_pos_xy=actual_grasp_pos[:2],
                )

                classifier_succ = (
                    "custom_events" in all_infos[-1]
                    and "E_SUCCESS"
                    in map(lambda x: x[1], all_infos[-1]["custom_events"])
                )

                if termination:
                    print(
                        f"[eval] Episode {self.episode_num} terminated "
                        f"(len={episode_length}, cycle={cycle_time_s:.2f}s, "
                        f"classifier={'OK' if classifier_succ else 'FAIL'})."
                    )
                elif truncation:
                    print(
                        f"[eval] Episode {self.episode_num} truncated "
                        f"(len={episode_length}, cycle={cycle_time_s:.2f}s, "
                        f"classifier={'OK' if classifier_succ else 'FAIL'})."
                    )

                user_succ = self._ask_user_success(self.episode_num)

                ep_record = self._make_episode_record(
                    ft_log,
                    cycle_time_s,
                    episode_length,
                    classifier_succ,
                    user_succ,
                    current_reset_info,
                    current_start_cart,
                )
                self.episodes_data.append(ep_record)
                self._flush()

                pd = ep_record["pose_diff"]
                logging.info(
                    f"[eval] ep {self.episode_num}: classifier={classifier_succ}, "
                    f"user={user_succ}, "
                    f"F_mean={ep_record['force_N']['mean']}, "
                    f"F_max={ep_record['force_N']['max']}, "
                    f"T_mean={ep_record['torque_Nm']['mean']}, "
                    f"T_max={ep_record['torque_Nm']['max']}, "
                    f"cycle={cycle_time_s:.2f}s | "
                    f"start→GT_xy={pd.get('start_to_gt_goal_xy_mm')} mm  "
                    f"start→EST_xy={pd.get('start_to_est_goal_xy_mm')} mm  "
                    f"EST→GT_xy={pd.get('est_to_gt_goal_xy_mm')} mm  "
                    f"start_yaw={pd.get('start_yaw_deg')} deg  "
                    f"est_yaw={pd.get('est_goal_yaw_deg')} deg"
                )

                if data_queue is not None:
                    try:
                        data_queue.put(
                            (all_actions, all_observations, all_rewards, termination)
                        )
                    except Exception:
                        pass

                self.episode_num += 1
                if max_eps > 0 and len(self.episodes_data) >= max_eps:
                    print(
                        f"[eval] reached target of {max_eps} valid episodes — stopping."
                    )
                    break

                obs_dict, reset_info = self.env.reset()
                start_cart = self._cartesian_from_obs(obs_dict)
                obs = obs_dict["observation.formatted"]
                actual_grasp_pos = reset_info["reset.grasped.position"]
                current_reset_info = reset_info
                current_start_cart = start_cart
                all_actions = []
                all_rewards = []
                all_observations = [obs]
                all_infos = [reset_info]
                ft_log = []
                episode_length = 0
                t_episode_start = time.perf_counter()

        except SystemExit:
            logging.info("[eval] quit requested. Saving partial results...")
        except KeyboardInterrupt:
            logging.info("[eval] keyboard interrupt. Saving partial results...")
        except Exception as e:
            logging.error(f"[eval] error in run_eval: {e}", exc_info=True)
        finally:
            try:
                self._flush()
                summary = self._compute_summary()
                print("\n=========== EVAL SUMMARY ===========")
                print(json.dumps(summary, indent=2))
                print(f"saved → {self.eval_json_path}")
                print("====================================\n")
            except Exception as e:
                logging.error(f"[eval] failed to flush JSON: {e}", exc_info=True)
            self.close()


# JSON schema (for reference):
#
# {
#   "meta": { "load_policy", "task", "success_threshold", "snap_reinforce",
#             "max_episodes", "prompt_user", "n_cameras", "max_action",
#             "started_at", "now", ... },
#   "episodes": [
#     {
#       "episode": int,
#       "datetime": "YYYY-mm-dd HH:MM:SS",
#       "length_steps": int,
#       "cycle_time_s": float,
#       "user_success": bool | null,
#       "classifier_success": bool,
#       "force_N":   {"mean", "max", "fx_mean", "fy_mean", "fz_mean",
#                     "fx_max_abs", "fy_max_abs", "fz_max_abs"},
#       "torque_Nm": {"mean", "max", "tx_mean", "ty_mean", "tz_mean",
#                     "tx_max_abs", "ty_max_abs", "tz_max_abs"},
#       "reset.grasped.delta"?: [...],
#       "reset.grasped.delta_estimated"?: [...],
#       "reset.goal_position.offset"?: [...]
#     }, ...
#   ],
#   "summary": {
#     "n_episodes", "n_unusable_skipped",
#     "success_rate_user", "n_user_labelled",
#     "success_rate_classifier", "agreement_rate",
#     "force_N":   {"episode_mean_stats": {mean,std,max},
#                   "episode_max_stats":  {mean,std,max}},
#     "torque_Nm": {"episode_mean_stats": {mean,std,max},
#                   "episode_max_stats":  {mean,std,max}},
#     "cycle_time_s": {mean, std, max}
#   }
# }
