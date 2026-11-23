"""多进程 Actor-Learner 并行环境池（路线 1：零额外依赖，直接压榨多核 CPU）。

设计原理：
- 启动 `n_workers` 个独立子进程，每个子进程负责各自一批 `MahjongEnv` 的规则步进、
  `ObsEncoder` 编码、`encode_event_stream` 事件流提取与 `fast_cache` 向听计算；
- 主进程（Learner / GPU）每步收集所有子进程返回的待决策状态，拼成一个大 batch
  做一次集中神经网络前向推理，再把采样好的 `(slot, old_logp)` 分发回子进程继续推进。

用法：
    with ParallelDealPool(n_workers=4, obs_order=False, event_len=48) as pool:
        outs = pool.run_deals(hstates, seeds, policy_fn=batch_infer)
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import numpy as np

from mahjong import env as E
from mahjong import event_stream as ES
from mahjong import fast_cache as FC
from mahjong import obs2 as O2
from mahjong import scenarios as SC
from mahjong import touch as TOUCH
from mahjong.hanchan import HanchanState

N_ACTIONS = 47
SLOT_TSUMO = 34
SLOT_RON = 35
SLOT_RIICHI = 36
SLOT_RYUUKYOKU = 37
SLOT_NAGASHI = 38
SLOT_PON = 39
SLOT_MINKAN = 40
SLOT_ADD_KAN = 41
SLOT_ANKAN = 42
SLOT_PASS = 43
SLOT_CHI_BASE = 44

_SLOTS = {
    E.ACT_TSUMO: SLOT_TSUMO,
    E.ACT_RON: SLOT_RON,
    E.ACT_RIICHI: SLOT_RIICHI,
    E.ACT_RYUUKYOKU: SLOT_RYUUKYOKU,
    E.ACT_NAGASHI: SLOT_NAGASHI,
    E.ACT_PON: SLOT_PON,
    E.ACT_MINKAN: SLOT_MINKAN,
    E.ACT_ADD_KAN: SLOT_ADD_KAN,
    E.ACT_ANKAN: SLOT_ANKAN,
    E.ACT_PASS: SLOT_PASS,
}


def _slot_of(action) -> int:
    t = action.type
    if t == E.ACT_DISCARD and 0 <= action.tile < 34:
        return action.tile
    if t == E.ACT_CHI:
        return SLOT_CHI_BASE + (action.tile % 3)
    return _SLOTS.get(t, SLOT_PASS)


def _filter_legal(legal, allow_ryuukyoku: bool):
    if allow_ryuukyoku or not legal:
        return legal
    filtered = [a for a in legal if a.type not in (E.ACT_RYUUKYOKU, E.ACT_NAGASHI)]
    return filtered if filtered else legal


def _encode_fast(env, encoder, hstate, p_idx, event_len: int):
    planes, scalar = encoder.encode(env, p_idx)
    c_tuple = tuple(env.players[p_idx].full_counts())
    sh = FC.cached_shanten(c_tuple)
    sit_vec, sit, give_up = SC.situation_features(env, hstate, p_idx, sh)
    vec = np.concatenate([planes.ravel(), scalar, sit_vec]).astype(np.float32)
    melds = len(env.players[p_idx].melds)
    waits = FC.fast_wait_count(c_tuple) if sh == 0 else 0
    tf = TOUCH.shape_features(c_tuple, melds, sh, waits)
    ev = ES.encode_event_stream(env, p_idx, max_len=event_len)
    return vec, sh, sit.label, give_up, tf, ev


def _after_discard_shanten(env, p_idx, act) -> int:
    if env.phase != E.Phase.DISCARD:
        return 8
    pl = env.players[p_idx]
    if act.tile >= 0 and act.tile in pl.hand:
        pl.hand.remove(act.tile)
        pl.hand.sort()
        sa = FC.cached_shanten(tuple(pl.full_counts()))
        pl.hand.append(act.tile)
        pl.hand.sort()
        return sa
    return 8


def _worker_loop(conn, obs_order: bool, event_len: int, allow_ryuukyoku: bool):
    """子进程主循环：接收一批 (hstate_dict, seed)，逐步与主进程交互完成一轮 deal。"""
    encoder = O2.ObsEncoder(order=obs_order)
    while True:
        msg = conn.recv()
        if msg is None or msg[0] == "CLOSE":
            break
        if msg[0] != "RUN_DEALS":
            continue

        _, hstate_specs, seeds = msg
        n_envs = len(seeds)
        hstates = []
        envs = []
        for spec, seed in zip(hstate_specs, seeds):
            h = HanchanState(dealer=spec["dealer"], start_points=25000)
            h.sticks = list(spec["sticks"])
            h.honba = spec["honba"]
            h.kyokuten = spec["kyokuten"]
            h.round = spec["round"]
            h.east_only = spec["east_only"]
            h.kyoku = spec["kyoku"]
            h.finished = spec["finished"]
            hstates.append(h)
            envs.append(h.new_env(seed=seed))

        infos = [[] for _ in range(n_envs)]
        results = [env.start_turn() for env in envs]
        guards = [0] * n_envs

        while True:
            active = [i for i in range(n_envs) if results[i] is None and guards[i] < 400]
            if not active:
                break

            pending_meta = []
            vecs, tfs, evs, slot_lists = [], [], [], []
            actions_by_env = {i: {} for i in active}

            for i in active:
                guards[i] += 1
                env = envs[i]
                hstate = hstates[i]
                for p in env.pending_players():
                    legal = _filter_legal(env.legal_actions(p), allow_ryuukyoku)
                    if not legal:
                        actions_by_env[i][p] = E.PASS_ACTION
                        continue
                    vec, sh, sit_label, give_up, tf, ev = _encode_fast(
                        env, encoder, hstate, p, event_len
                    )
                    slots = [_slot_of(a) for a in legal]
                    vecs.append(vec)
                    tfs.append(tf)
                    evs.append(ev)
                    slot_lists.append(slots)
                    pending_meta.append((i, p, sh, sit_label, give_up, legal, slots, vec, tf, ev))

            if pending_meta:
                conn.send((
                    "INFER",
                    np.stack(vecs),
                    np.stack(tfs),
                    np.stack(evs),
                    slot_lists,
                ))
                chosen = conn.recv()  # list of (pick_slot, old_logp, best_riichi_tile_or_neg1)

                for meta, (pick_slot, old_lp, riichi_tile) in zip(pending_meta, chosen):
                    i, p, sh, sit_label, give_up, legal, slots, vec, tf, ev = meta
                    cands = [a for s, a in zip(slots, legal) if s == pick_slot]
                    if not cands:
                        act = legal[0]
                    elif pick_slot == SLOT_RIICHI and len(cands) > 1 and riichi_tile >= 0:
                        matched = [a for a in cands if a.tile == riichi_tile]
                        act = matched[0] if matched else cands[0]
                    else:
                        act = cands[0]
                    actions_by_env[i][p] = act
                    sa = _after_discard_shanten(envs[i], p, act)
                    infos[i].append(
                        (vec, slots, pick_slot, sh, sa, p, sit_label, give_up, tf, old_lp, ev)
                    )

            for i in active:
                results[i] = envs[i].step(actions_by_env[i])

        conn.send(("DONE", [(envs[i], results[i], infos[i]) for i in range(n_envs)]))


class ParallelDealPool:
    """管理多进程环境 Worker，并将所有 Worker 的推理请求合并成单次主进程 batch 前向。"""

    def __init__(
        self,
        n_workers: int = 4,
        obs_order: bool = False,
        event_len: int = ES.DEFAULT_EVENT_LEN,
        allow_ryuukyoku: bool = False,
    ):
        self.n_workers = max(1, int(n_workers))
        self.obs_order = bool(obs_order)
        self.event_len = int(event_len)
        self.allow_ryuukyoku = bool(allow_ryuukyoku)
        self._procs = []
        self._conns = []

    def start(self) -> "ParallelDealPool":
        if self._procs:
            return self
        ctx = mp.get_context("spawn")
        for _ in range(self.n_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker_loop,
                args=(child_conn, self.obs_order, self.event_len, self.allow_ryuukyoku),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._procs.append(p)
            self._conns.append(parent_conn)
        return self

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("CLOSE",))
                conn.close()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        self._procs.clear()
        self._conns.clear()

    def __enter__(self) -> "ParallelDealPool":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run_deals(
        self,
        hstates: list[HanchanState],
        seeds: list[int],
        policy_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, list[list[int]]], list[tuple[int, float, int]]],
    ):
        """把一批 deal 均分给所有 Worker，每轮聚合所有 Worker 的待推理状态一次性调用 policy_fn。"""
        if not self._procs:
            self.start()
        n = len(seeds)
        if n == 0:
            return []

        active_workers = min(self.n_workers, n)
        buckets = [[] for _ in range(active_workers)]
        for idx in range(n):
            buckets[idx % active_workers].append(idx)

        for w_idx in range(active_workers):
            idxs = buckets[w_idx]
            specs = [
                {
                    "dealer": hstates[i].dealer,
                    "sticks": list(hstates[i].sticks),
                    "honba": hstates[i].honba,
                    "kyokuten": hstates[i].kyokuten,
                    "round": hstates[i].round,
                    "east_only": hstates[i].east_only,
                    "kyoku": hstates[i].kyoku,
                    "finished": hstates[i].finished,
                }
                for i in idxs
            ]
            w_seeds = [seeds[i] for i in idxs]
            self._conns[w_idx].send(("RUN_DEALS", specs, w_seeds))

        done_flags = [False] * active_workers
        outputs = [None] * n

        while not all(done_flags):
            infer_workers = []
            infer_vecs, infer_tfs, infer_evs, infer_slots = [], [], [], []
            infer_counts = []

            for w_idx in range(active_workers):
                if done_flags[w_idx]:
                    continue
                msg = self._conns[w_idx].recv()
                tag = msg[0]
                if tag == "DONE":
                    done_flags[w_idx] = True
                    for local_k, orig_i in enumerate(buckets[w_idx]):
                        outputs[orig_i] = msg[1][local_k]
                elif tag == "INFER":
                    _, v_arr, tf_arr, ev_arr, sl_list = msg
                    infer_workers.append(w_idx)
                    infer_vecs.append(v_arr)
                    infer_tfs.append(tf_arr)
                    infer_evs.append(ev_arr)
                    infer_slots.extend(sl_list)
                    infer_counts.append(len(sl_list))

            if infer_workers:
                big_v = np.concatenate(infer_vecs, axis=0)
                big_tf = np.concatenate(infer_tfs, axis=0)
                big_ev = np.concatenate(infer_evs, axis=0)
                all_chosen = policy_fn(big_v, big_tf, big_ev, infer_slots)

                offset = 0
                for w_idx, cnt in zip(infer_workers, infer_counts):
                    self._conns[w_idx].send(all_chosen[offset : offset + cnt])
                    offset += cnt

        return outputs
