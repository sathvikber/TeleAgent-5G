import re
import subprocess
from typing import Dict, Optional


class RouteResolver:
    """Discovers the active egress interface and source IP for a target."""

    @staticmethod
    def resolve(target_ip: str) -> Dict[str, Optional[str]]:
        try:
            res = subprocess.run(
                ["ip", "route", "get", target_ip],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if res.returncode != 0:
                return {"interface": None, "src_ip": None, "route_found": False}

            output = res.stdout.strip()
            dev_match = re.search(r"dev\s+(\S+)", output)
            src_match = re.search(r"src\s+(\S+)", output)

            return {
                "interface": dev_match.group(1) if dev_match else None,
                "src_ip": src_match.group(1) if src_match else None,
                "route_found": True,
            }
        except Exception as e:
            return {
                "interface": None,
                "src_ip": None,
                "route_found": False,
                "error": str(e),
            }
