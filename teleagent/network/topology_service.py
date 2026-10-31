"""
Lab Topology Service

Manages real-time physical/logical 5G network topology states,
interface connections, and topology delta logs.
"""

import time
from typing import Dict, List, Any, Optional


class LabTopologyService:
    def __init__(self, config_path: Optional[str] = None):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._links: Dict[str, Dict[str, Any]] = {}
        self._state_changes: List[Dict[str, Any]] = []
        self._initialize_default_topology()

    def _initialize_default_topology(self) -> None:
        default_nodes = [
            ("amf-01", "AMF", "192.168.10.10"),
            ("smf-01", "SMF", "192.168.10.11"),
            ("upf-main", "UPF", "192.168.10.12"),
            ("ausf-01", "AUSF", "192.168.10.13"),
            ("udm-01", "UDM", "192.168.10.14"),
            ("gnb-01", "gNodeB", "192.168.10.20"),
            ("gnb-02", "gNodeB", "192.168.10.21"),
        ]
        for node_id, role, ip in default_nodes:
            self.register_node(node_id, role, ip)

        default_links = [
            ("link-gnb1-amf", "gnb-01", "amf-01", "N2", "UP"),
            ("link-gnb2-amf", "gnb-02", "amf-01", "N2", "UP"),
            ("link-gnb1-upf", "gnb-01", "upf-main", "N3", "UP"),
            ("link-gnb2-upf", "gnb-02", "upf-main", "N3", "UP"),
            ("link-smf-upf", "smf-01", "upf-main", "N4", "UP"),
            ("link-amf-smf", "amf-01", "smf-01", "N11", "UP"),
        ]
        for link_id, src, tgt, interface_type, status in default_links:
            self.register_link(link_id, src, tgt, interface_type, status)

    def register_node(self, node_id: str, role: str, ip_address: str, status: str = "UP") -> None:
        self._nodes[node_id] = {
            "node_id": node_id,
            "role": role,
            "ip_address": ip_address,
            "status": status,
        }

    def register_link(self, link_id: str, source: str, target: str, interface_type: str, status: str = "UP") -> None:
        self._links[link_id] = {
            "link_id": link_id,
            "source": source,
            "target": target,
            "interface_type": interface_type,
            "status": status,
        }

    def update_link_status(self, link_id: str, new_status: str, reason: str = "") -> None:
        if link_id in self._links:
            old_status = self._links[link_id]["status"]
            if old_status != new_status:
                self._links[link_id]["status"] = new_status
                change_event = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "link_id": link_id,
                    "interface": self._links[link_id]["interface_type"],
                    "source": self._links[link_id]["source"],
                    "target": self._links[link_id]["target"],
                    "old_status": old_status,
                    "new_status": new_status,
                    "reason": reason,
                }
                self._state_changes.append(change_event)

    def get_topology_summary(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self._nodes),
            "total_links": len(self._links),
            "active_nodes": [f"{n['node_id']} ({n['role']})" for n in self._nodes.values() if n["status"] == "UP"],
            "degraded_links": [
                f"{l['link_id']} [{l['interface_type']}] ({l['source']} -> {l['target']}) : {l['status']}"
                for l in self._links.values() if l["status"] != "UP"
            ],
            "recent_state_changes": self._state_changes[-5:],
        }

    def get_active_node_ips(self) -> List[str]:
        return [node["ip_address"] for node in self._nodes.values() if "ip_address" in node]