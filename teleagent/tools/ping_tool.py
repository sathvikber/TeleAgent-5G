import asyncio
import platform
import re
import subprocess
from typing import Optional, Tuple


class Prober:
    """Performs non-blocking ICMP and TCP probes."""

    @staticmethod
    async def icmp_ping(
        target_ip: str, timeout: float = 1.0
    ) -> Tuple[bool, Optional[float]]:
        system = platform.system().lower()
        timeout_arg = str(int(timeout * 1000)) if system == "darwin" else str(int(timeout))
        
        try:
            proc = await asyncio.create_subprocess_exec(
                "ping",
                "-c",
                "1",
                "-W",
                timeout_arg,
                target_ip,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout + 0.5
            )
            if proc.returncode == 0:
                match = re.search(r"time=([\d.]+)\s*ms", stdout.decode())
                rtt = float(match.group(1)) if match else 0.0
                return True, rtt
            return False, None
        except Exception:
            return False, None

    @staticmethod
    async def tcp_connect(
        target_ip: str, port: int, timeout: float = 1.0
    ) -> Tuple[bool, Optional[float]]:
        start = asyncio.get_event_loop().time()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(target_ip, port), timeout=timeout
            )
            rtt = (asyncio.get_event_loop().time() - start) * 1000.0
            writer.close()
            await writer.wait_closed()
            return True, round(rtt, 2)
        except Exception:
            return False, None
