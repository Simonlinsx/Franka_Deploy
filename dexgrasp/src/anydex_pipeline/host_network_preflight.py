"""Read-only host network gates for formal Franka execution.

The FCI command stream has a one-millisecond end-to-end deadline.  A browser
connected to Desk on the same dedicated robot link is not motion authority and
must not share that link with a formal control run.  This module inspects the
kernel's TCP socket tables directly; it never opens a socket, sends a packet,
terminates a process, or changes host networking.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
from typing import Iterable, List, Tuple, Union


TCP_ESTABLISHED = "01"
FRANKA_DESK_HTTPS_PORT = 443


class HostNetworkPreflightError(RuntimeError):
    """The host TCP state could not prove an uncontended Franka link."""


@dataclass(frozen=True)
class EstablishedTcpConnection:
    """One established TCP connection decoded from ``/proc/net/tcp*``."""

    local_ip: str
    local_port: int
    remote_ip: str
    remote_port: int
    kernel_table: str


def _decode_proc_ipv4(value: str) -> ipaddress.IPv4Address:
    raw = bytes.fromhex(value)
    if len(raw) != 4:
        raise ValueError("IPv4 address must contain four bytes")
    return ipaddress.IPv4Address(raw[::-1])


def _decode_proc_ipv6(value: str) -> ipaddress.IPv6Address:
    raw = bytes.fromhex(value)
    if len(raw) != 16:
        raise ValueError("IPv6 address must contain sixteen bytes")
    # /proc/net/tcp6 prints each native-endian 32-bit word independently.
    network_order = b"".join(
        raw[offset : offset + 4][::-1] for offset in range(0, 16, 4)
    )
    return ipaddress.IPv6Address(network_order)


IpAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]


def _decode_endpoint(value: str, *, ipv6: bool) -> Tuple[IpAddress, int]:
    try:
        address_hex, port_hex = value.rsplit(":", 1)
        address = (
            _decode_proc_ipv6(address_hex)
            if ipv6
            else _decode_proc_ipv4(address_hex)
        )
        port = int(port_hex, 16)
    except (TypeError, ValueError) as exc:
        raise HostNetworkPreflightError(
            "malformed endpoint in kernel TCP table: {!r}".format(value)
        ) from exc
    if not 0 <= port <= 65535:
        raise HostNetworkPreflightError(
            "invalid port in kernel TCP table: {!r}".format(value)
        )
    return address, port


def _normalized_ip(value: IpAddress) -> IpAddress:
    if isinstance(value, ipaddress.IPv6Address) and value.ipv4_mapped is not None:
        return value.ipv4_mapped
    return value


def _parse_tcp_table(
    lines: Iterable[str], *, ipv6: bool, table_name: str
) -> Tuple[EstablishedTcpConnection, ...]:
    connections: List[EstablishedTcpConnection] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields or fields[0] == "sl":
            continue
        if len(fields) < 4:
            raise HostNetworkPreflightError(
                "malformed {} line {}: expected at least four fields".format(
                    table_name, line_number
                )
            )
        if fields[3] != TCP_ESTABLISHED:
            continue
        local_ip, local_port = _decode_endpoint(fields[1], ipv6=ipv6)
        remote_ip, remote_port = _decode_endpoint(fields[2], ipv6=ipv6)
        connections.append(
            EstablishedTcpConnection(
                local_ip=str(_normalized_ip(local_ip)),
                local_port=local_port,
                remote_ip=str(_normalized_ip(remote_ip)),
                remote_port=remote_port,
                kernel_table=table_name,
            )
        )
    return tuple(connections)


def established_tcp_connections_to(
    remote_ip: str,
    remote_port: int = FRANKA_DESK_HTTPS_PORT,
    *,
    proc_net_dir: Path = Path("/proc/net"),
) -> Tuple[EstablishedTcpConnection, ...]:
    """Return established TCP connections to one literal IP and port.

    Both IPv4 and IPv6 kernel tables are required.  An unreadable or malformed
    table is a failed inspection, not evidence that the link is uncontended.
    """

    try:
        wanted_ip = _normalized_ip(ipaddress.ip_address(str(remote_ip).strip()))
    except ValueError as exc:
        raise HostNetworkPreflightError(
            "Franka host preflight requires a literal robot IP address, "
            "got {!r}".format(remote_ip)
        ) from exc
    try:
        wanted_port = int(remote_port)
    except (TypeError, ValueError) as exc:
        raise HostNetworkPreflightError(
            "Franka host preflight port is invalid: {!r}".format(remote_port)
        ) from exc
    if not 1 <= wanted_port <= 65535:
        raise HostNetworkPreflightError(
            "Franka host preflight port is outside 1..65535: {}".format(
                wanted_port
            )
        )

    all_connections: List[EstablishedTcpConnection] = []
    for filename, ipv6 in (("tcp", False), ("tcp6", True)):
        path = Path(proc_net_dir) / filename
        try:
            with path.open("r", encoding="ascii") as stream:
                all_connections.extend(
                    _parse_tcp_table(
                        stream,
                        ipv6=ipv6,
                        table_name=str(path),
                    )
                )
        except HostNetworkPreflightError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HostNetworkPreflightError(
                "cannot inspect kernel TCP table {}: {}".format(path, exc)
            ) from exc

    return tuple(
        connection
        for connection in all_connections
        if _normalized_ip(ipaddress.ip_address(connection.remote_ip)) == wanted_ip
        and connection.remote_port == wanted_port
    )


def require_uncontended_franka_https_link(
    robot_ip: str,
    *,
    proc_net_dir: Path = Path("/proc/net"),
) -> None:
    """Fail if Desk/browser HTTPS is established to the Franka controller."""

    connections = established_tcp_connections_to(
        robot_ip,
        FRANKA_DESK_HTTPS_PORT,
        proc_net_dir=proc_net_dir,
    )
    if not connections:
        return
    local_endpoints = sorted(
        {"{}:{}".format(item.local_ip, item.local_port) for item in connections}
    )
    preview = ", ".join(local_endpoints[:4])
    if len(local_endpoints) > 4:
        preview += ", ..."
    raise HostNetworkPreflightError(
        "motion blocked: found {} ESTABLISHED TCP connection(s) to Franka "
        "Desk at {}:{} (local {}). Close every Desk/Chrome tab or window "
        "connected to this robot, wait for the ESTABLISHED connections to "
        "close, and rerun. This check did not terminate any process or "
        "change host networking.".format(
            len(connections),
            robot_ip,
            FRANKA_DESK_HTTPS_PORT,
            preview,
        )
    )
