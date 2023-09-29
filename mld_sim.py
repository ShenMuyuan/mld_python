# Author: Muyuan Shen <shmy315@outlook.com>
# Simulation for non-STR MLDs (synchronous)
# 1) Saturated: longest backoff (LB) or shortest backoff (SB)
# 2) Unsaturated (Bernoulli arrival): choose from policies TBD

import random
from functools import reduce
from math import ceil
import numpy as np
from enum import IntEnum, auto

# macro definitions
# link state
class LinkState(IntEnum):
    LINK_IDLE = auto()
    LINK_SUCC = auto()
    LINK_COLL = auto()
# sat backoff type
class SatBackoffType(IntEnum):
    BO_SHORTEST = auto()
    BO_LONGEST = auto()
# unsat policy type
class UnsatPolicyType(IntEnum):
    POL_ONE = auto()
    POL_TWO = auto()
    POL_THREE = auto()
# event type
class EventType(IntEnum):   # smaller enum has higher priority
    EV_SUCC_END = auto()
    EV_COLL_END = auto()
    EV_BO = auto()
    # EV_SUCC_START = auto()
    # EV_COLL_START = auto()

# time parameters
slot_us = 9
sifs_us = 16
difs_us = sifs_us + 2 * slot_us
phy_preamble_us = 20

# size parameters (bits)
## payload_len (variable)
mac_header_len = 288
ack_len = 112

# rate parameters (Mbps)
## basic_rate (variable)
## tx_rate (variable)

# DCF parameters
## init_cw (variable)
## cutoff_phase (variable)

# topology parameters
## n_sta (variable)
## n_link (variable)

# other parameters
## backoff_type (variable)


def calculate_succ_holding_time_us(payload_len, tx_rate_mbps, basic_rate_mbps):
    tx_us = (payload_len + mac_header_len) / tx_rate_mbps * 1000
    ack_us = ack_len / basic_rate_mbps * 1000
    return round(phy_preamble_us + tx_us + sifs_us + ack_us + difs_us)


def calculate_coll_holding_time_us(payload_len, tx_rate_mbps):
    tx_us = (payload_len + mac_header_len) / tx_rate_mbps * 1000
    return round(phy_preamble_us + tx_us + difs_us)


class MLD_sta:

    def __init__(self, n_link, sat, sat_bo_type, unsat_pol_type, unsat_arrival_rate, tx_rate, payload_len, basic_rate, init_cw, cutoff_phase):

        # constants
        self.n_link = n_link
        self.sat = sat
        assert type(sat) is bool
        if self.sat:
            assert (sat_bo_type == SatBackoffType.BO_LONGEST) or (sat_bo_type == SatBackoffType.BO_SHORTEST)
            self.sat_bo_type = sat_bo_type
            self.unsat_pol_type = None
            self.arrival_rate = None
        else:
            self.sat_bo_type = None
            assert (unsat_pol_type == UnsatPolicyType.POL_ONE) or (unsat_pol_type == UnsatPolicyType.POL_TWO) or (unsat_pol_type == UnsatPolicyType.POL_THREE)
            self.unsat_pol_type = unsat_pol_type
            assert (unsat_arrival_rate > 0) and (unsat_arrival_rate <= 1)
            self.arrival_rate = unsat_arrival_rate
        self.tx_rate = tx_rate
        self.payload_len = payload_len
        self.basic_rate = basic_rate
        self.init_cw = init_cw
        self.cutoff_phase = cutoff_phase

        # holding times
        self.succ_hold_us = calculate_succ_holding_time_us(self.payload_len, self.tx_rate, self.basic_rate)
        self.coll_hold_us = calculate_coll_holding_time_us(self.payload_len, self.tx_rate)
        self.bernoulli_arrival_prob_per_slot = self.arrival_rate / ceil(self.succ_hold_us / slot_us)

        # variables
        self.queue_len = None
        self.cw = None
        self.phase = None
        self.count = None
        self.dirty = False

    def reset_all_backoff(self):
        self.count = []
        for _ in range(self.n_link):
            self.count.append(random.randrange(self.cw * (2 ** self.phase)))

    def bernoulli_arrival(self):
        self.queue_len += 1

    def handle_collision(self):
        if self.phase < self.cutoff_phase:
            self.phase += 1
        self.reset_all_backoff()

    def handle_success(self):
        self.phase = 0
        self.reset_all_backoff()

    def get_bernoulli_arrival(self):
        assert not self.sat
        if random.random() < self.bernoulli_arrival_prob_per_slot:
            self.queue_len += 1

    def step(self, link_states):
        if not self.dirty:
            if not self.sat:
                self.queue_len = 0
            self.cw = self.init_cw
            self.phase = 0
            self.reset_all_backoff()
            self.dirty = True
        n_idles = 0
        for i in range(self.n_link):
            if link_states[i][0] == LinkState.LINK_IDLE:
                n_idles += 1
                if self.count[i] > 0:
                    self.count[i] -= 1
        if n_idles != self.n_link:
            return np.zeros((self.n_link,), dtype=int), self.succ_hold_us, self.coll_hold_us
        if self.sat:
            want_links = None
            if self.sat_bo_type == SatBackoffType.BO_LONGEST:
                if sum(self.count) == 0:
                    want_links = np.ones((self.n_link,), dtype=int)
                else:
                    want_links = np.zeros((self.n_link,), dtype=int)
            if self.sat_bo_type == SatBackoffType.BO_SHORTEST:
                if reduce(lambda x, y: x * y, self.count) == 0:
                    want_links = np.ones((self.n_link,), dtype=int)
                else:
                    want_links = np.zeros((self.n_link,), dtype=int)
        else:
            self.get_bernoulli_arrival()
            want_links = None
            match self.unsat_pol_type:
                case UnsatPolicyType.POL_ONE:
                    if (self.queue_len > 0) and (reduce(lambda x, y: x * y, self.count) == 0):
                        # want to send on links that count=0 as more as possible
                        want_links = np.zeros((self.n_link,), dtype=int)
                        avail_packet_num = self.queue_len
                        for i in range(self.n_link):
                            if avail_packet_num == 0:
                                break
                            if self.count[i] == 0:
                                avail_packet_num -= 1
                                want_links[i] = 1
                case UnsatPolicyType.POL_TWO:
                    if (self.queue_len >= self.n_link) and (sum(self.count) == 0):
                        # want to send on all links when all links count=0
                        want_links = np.ones((self.n_link,), dtype=int)
                case UnsatPolicyType.POL_THREE:
                    if (self.queue_len > 0) and (reduce(lambda x, y: x * y, self.count) == 0):
                        # want to send on links as more as possible
                        want_links = np.zeros((self.n_link,), dtype=int)
                        avail_packet_num = self.queue_len
                        for i in range(avail_packet_num):
                            want_links[i] = 1
        assert want_links is not None
        return want_links, self.succ_hold_us, self.coll_hold_us


class MLD_experiment_non_STR:

    def __init__(self, n_sta, n_link, sat, backoff_type, unsat_pol_type, unsat_arrival_rate, tx_rate, payload_len,
                 basic_rate=24, init_cw=16, cutoff_phase=6):
        self.n_sta = n_sta
        self.n_link = n_link
        self.sat = sat
        self.backoff_type = backoff_type
        self.unsat_pol_type = unsat_pol_type
        self.unsat_arrival_rate = unsat_arrival_rate
        self.tx_rate = tx_rate
        self.payload_len = payload_len
        self.basic_rate = basic_rate
        self.init_cw = init_cw
        self.cutoff_phase = cutoff_phase

        # STAs
        self.mld_stas = [MLD_sta(self.n_link, self.sat, self.backoff_type, self.unsat_pol_type, self.unsat_arrival_rate,
                                 self.tx_rate, self.payload_len, self.basic_rate, self.init_cw, self.cutoff_phase)
                         for i in range(self.n_sta)]

        # link states
        # node id: -1=invalid, 0~(n_link-1)=STA
        self.link_states = [[LinkState.LINK_IDLE, -1]] * n_link  # [link state, tx node id (-1 if link idle)]

        # events
        self.upcoming_events = None
        self.all_events = None

        # statistics
        self.current_time_us = None
        self.success_frames = None
        self.success_data = None
        self.collide_count = None
        self.dirty = False

    def update_link_state(self, link, state, args=None):
        if args is None:
            args = -1
        self.link_states[link] = [state, args]


    def add_one_event(self, interval, ev_type, ev_arg):
        # should add to both upcoming and all
        assert self.upcoming_events is not None
        assert self.all_events is not None
        self.upcoming_events = np.vstack((self.upcoming_events, [self.current_time_us + interval, ev_type, ev_arg]))
        pass

    def process_one_event(self):
        if not self.dirty:
            assert self.upcoming_events is None
            assert self.all_events is None
            assert self.current_time_us is None
            assert self.success_frames is None
            assert self.success_data is None
            assert self.collide_count is None
            self.upcoming_events = np.array([[0, EventType.EV_BO, 0]])
            self.all_events = np.array([[0, EventType.EV_BO, 0]])
            self.current_time_us = 0
            self.success_frames = 0
            self.success_data = 0
            self.collide_count = 0
            self.dirty = True

        # find the event with highest priority (smallest in enum) and earliest time, and dispose the event
        event_index = np.lexsort((self.upcoming_events[:, 1], self.upcoming_events[:, 0]))[0]
        event = self.upcoming_events[event_index]
        self.upcoming_events = np.delete(self.upcoming_events, event_index, axis=0)
        self.current_time_us = event[0]
        match event[1]:
            case EventType.EV_BO:
                self.do_backoff()
            case EventType.EV_COLL_END:
                self.mld_stas[event[2]].handle_collision()




    def do_backoff(self):
        requirements_table = np.zeros((self.n_link, self.n_sta), dtype=int)
        succ_hold_table = np.zeros(self.n_sta)
        coll_hold_table = np.zeros(self.n_sta)
        for i in range(self.n_sta):
            want_links, succ_hold_us, coll_hold_us = self.mld_stas[i].step(self.link_states)
            requirements_table[:, i] = want_links
            succ_hold_table[i] = succ_hold_us
            coll_hold_table[i] = coll_hold_us
        for i in range(self.n_link):
            if sum(requirements_table[i]) == 1:
                # success on that link
                succ_sta = np.nonzero(requirements_table[i])[0][0]
                self.add_one_event(succ_hold_table[succ_sta], EventType.EV_SUCC_END, succ_sta)
                self.update_link_state(i, LinkState.LINK_SUCC, succ_sta)
            elif sum(requirements_table[i]) > 1:
                # collision on that link
                coll_stas = np.nonzero(requirements_table[i])[0]
                max_coll_hold = 0
                for j in range(len(coll_stas)):
                    if coll_hold_table[coll_stas[j]] > max_coll_hold:
                        max_coll_hold = coll_hold_table[coll_stas[j]]
                assert max_coll_hold != 0
                for j in range(len(coll_stas)):
                    self.add_one_event(max_coll_hold, EventType.EV_COLL_END, coll_stas[j])
                    self.update_link_state(i, LinkState.LINK_COLL, coll_stas[j])


    def check_collision(self):
        pass

    def tick(self):
        for i in range(self.n_sta + 1):
            self.step(i)
        self.check_collision()
        self.current_time_us += 1
        pass





