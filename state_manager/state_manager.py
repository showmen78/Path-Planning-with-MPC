"""
State manager package module.

Tracks the latest and recent historical snapshots for all simulation objects so
MPC, the tracker, and diagnostics read a consistent view of the world state.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Deque, Dict, List, Sequence

from vehicle_manager.vehicle import Vehicle


class StateManager:
    """
    Intent:
        Maintain current and historical snapshots for ego and non-ego objects.

    Method used:
        Per simulation step, read each object's snapshot and append it to a
        bounded history buffer (deque) indexed by `vehicle_id`.
    """

    def __init__(self, history_length: int = 300) -> None:
        if int(history_length) <= 0:
            raise ValueError("history_length must be > 0.")
        self.history_length = int(history_length)
        self.current_states: Dict[str, Dict[str, object]] = {}
        self.history: Dict[str, Deque[Dict[str, object]]] = {}
        self.last_timestamp_s: float | None = None

    def refresh(self, vehicles: Sequence[Vehicle], timestamp_s: float) -> Dict[str, Dict[str, object]]:
        """
        Intent:
            Refresh the tracked snapshot dictionaries from the current object list.
        """

        active_ids: set[str] = set()
        time_s = float(timestamp_s)
        for vehicle in vehicles:
            snapshot = vehicle.to_snapshot()
            object_id = str(snapshot["vehicle_id"])
            snapshot["timestamp_s"] = time_s
            active_ids.add(object_id)
            self.current_states[object_id] = snapshot
            if object_id not in self.history:
                self.history[object_id] = deque(maxlen=self.history_length)
            self.history[object_id].append(snapshot)

        stale_ids = set(self.current_states.keys()) - active_ids
        for stale_id in stale_ids:
            self.current_states.pop(stale_id, None)
            self.history.pop(stale_id, None)

        self.last_timestamp_s = time_s
        return deepcopy(self.current_states)

    def get_all_current_states(self) -> Dict[str, Dict[str, object]]:
        return deepcopy(self.current_states)

    def get_ego_state(self) -> Dict[str, object] | None:
        for snapshot in self.current_states.values():
            if str(snapshot.get("type", "")).lower() == "ego":
                return deepcopy(snapshot)
        return None

    def get_non_ego_states(self) -> List[Dict[str, object]]:
        output: List[Dict[str, object]] = []
        for snapshot in self.current_states.values():
            if str(snapshot.get("type", "")).lower() != "ego":
                output.append(deepcopy(snapshot))
        return output

    def get_vehicle_history(self, vehicle_id: str) -> List[Dict[str, object]]:
        history = self.history.get(str(vehicle_id))
        if history is None:
            return []
        return deepcopy(list(history))
