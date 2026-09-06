from __future__ import annotations

from pathlib import Path

import pytest

from anydex_pipeline.host_network_preflight import (
    HostNetworkPreflightError,
    established_tcp_connections_to,
    require_uncontended_franka_https_link,
)


HEADER = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
    "retrnsmt   uid  timeout inode\n"
)


def _write_tables(tmp_path: Path, tcp_lines=(), tcp6_lines=()) -> Path:
    proc_net = tmp_path / "net"
    proc_net.mkdir()
    (proc_net / "tcp").write_text(
        HEADER + "".join(tcp_lines), encoding="ascii"
    )
    (proc_net / "tcp6").write_text(
        HEADER + "".join(tcp6_lines), encoding="ascii"
    )
    return proc_net


def _row(index: int, local: str, remote: str, state: str) -> str:
    return "{}: {} {} {} 00000000:00000000\n".format(
        index, local, remote, state
    )


def test_finds_only_established_https_to_exact_robot_ipv4(tmp_path):
    proc_net = _write_tables(
        tmp_path,
        tcp_lines=(
            _row(0, "010010AC:C350", "020010AC:01BB", "01"),
            _row(1, "010010AC:C351", "020010AC:01BB", "06"),
            _row(2, "010010AC:C352", "020010AC:0050", "01"),
            _row(3, "010010AC:C353", "030010AC:01BB", "01"),
        ),
    )

    connections = established_tcp_connections_to(
        "172.16.0.2", proc_net_dir=proc_net
    )

    assert len(connections) == 1
    assert connections[0].local_ip == "172.16.0.1"
    assert connections[0].local_port == 0xC350
    assert connections[0].remote_ip == "172.16.0.2"
    assert connections[0].remote_port == 443


def test_ipv4_mapped_tcp6_connection_is_also_blocked(tmp_path):
    proc_net = _write_tables(
        tmp_path,
        tcp6_lines=(
            _row(
                0,
                "0000000000000000FFFF0000010010AC:C350",
                "0000000000000000FFFF0000020010AC:01BB",
                "01",
            ),
        ),
    )

    connections = established_tcp_connections_to(
        "172.16.0.2", proc_net_dir=proc_net
    )

    assert len(connections) == 1
    assert connections[0].local_ip == "172.16.0.1"
    assert connections[0].remote_ip == "172.16.0.2"


def test_unreadable_or_malformed_kernel_table_fails_closed(tmp_path):
    proc_net = tmp_path / "net"
    proc_net.mkdir()
    (proc_net / "tcp").write_text(HEADER, encoding="ascii")
    with pytest.raises(HostNetworkPreflightError, match="tcp6"):
        established_tcp_connections_to("172.16.0.2", proc_net_dir=proc_net)

    (proc_net / "tcp6").write_text(
        HEADER + "0: not-an-endpoint also-bad 01\n", encoding="ascii"
    )
    with pytest.raises(HostNetworkPreflightError, match="malformed endpoint"):
        established_tcp_connections_to("172.16.0.2", proc_net_dir=proc_net)


def test_non_literal_robot_host_fails_closed_without_dns(tmp_path):
    proc_net = _write_tables(tmp_path)
    with pytest.raises(HostNetworkPreflightError, match="literal robot IP"):
        established_tcp_connections_to(
            "franka-control.local", proc_net_dir=proc_net
        )


def test_block_message_is_actionable_and_does_not_claim_process_mutation(tmp_path):
    proc_net = _write_tables(
        tmp_path,
        tcp_lines=(
            _row(0, "010010AC:C350", "020010AC:01BB", "01"),
            _row(1, "010010AC:C351", "020010AC:01BB", "01"),
        ),
    )

    with pytest.raises(HostNetworkPreflightError) as captured:
        require_uncontended_franka_https_link(
            "172.16.0.2", proc_net_dir=proc_net
        )

    message = str(captured.value)
    assert "2 ESTABLISHED" in message
    assert "172.16.0.2:443" in message
    assert "Close every Desk/Chrome tab" in message
    assert "did not terminate any process" in message


def test_empty_tables_pass_without_opening_a_socket(tmp_path):
    proc_net = _write_tables(tmp_path)
    require_uncontended_franka_https_link(
        "172.16.0.2", proc_net_dir=proc_net
    )
