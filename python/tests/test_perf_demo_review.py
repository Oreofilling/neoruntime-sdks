"""Offline review regressions: real SDK framing/iterator, mocked upstream IO."""
import asyncio
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/perf_demo'))
from app import PerfDemoApp, parse_args
from chains.a_subscribe import ChainA
from chains.b_keepfd import ChainB
from metrics import MetricsHub
from statusline import EncodedWatcher

from neoruntime_ipc_sdk import EncodedStreamClient, InferenceClient
from neoruntime_ipc_sdk.proto import inference_pb2


def app_args(*extra):
    return parse_args(['--model-path', '/models/test.hef', '--model-id', 'test-owned', *extra])


def capture(hub):
    events = []
    hub.emit = lambda kind, **fields: events.append(dict(type=kind, **fields))
    return events


def test_watcher_transient_close_error_does_not_poison_final_verdict():
    hub = MetricsHub()
    watcher = EncodedWatcher('sub', hub)
    client = MagicMock()
    client.close.side_effect = [OSError('boom'), None]
    watcher._close_client(client)  # one dead cycle fails its close
    assert watcher.cleanup_error == 'OSError'
    watcher._close_client(client)  # the next clean cycle clears it
    assert watcher.cleanup_error is None


def test_watcher_real_socket_eof_closes_reconnects_and_receives_new_packet():
    hub = MetricsHub()
    events = []
    packet_seen = threading.Event()
    watcher = EncodedWatcher('sub', hub)
    def emit(kind, **fields):
        events.append(dict(type=kind, **fields))
        if kind == 'encoded_packet':
            packet_seen.set()
            watcher.stop()
    hub.emit = emit
    first_sock, first_peer = socket.socketpair()
    first_peer.close()  # real EOF, SDK returns None instead of raising
    second_sock, second_peer = socket.socketpair()
    second_peer.sendall(struct.pack('<IBBQIIQ', 31, 0, 0, 100, 640, 384, 100) + b'x')
    clients = [EncodedStreamClient(stream_id='sub'), EncodedStreamClient(stream_id='sub')]
    clients[0]._sock, clients[1]._sock = first_sock, second_sock
    try:
        with patch('neoruntime_ipc_sdk.EncodedStreamClient', side_effect=clients) as factory:
            watcher.start()
            assert packet_seen.wait(2), 'EOF must rebuild rather than spin forever on the dead socket'
            watcher.join(1)
            assert not watcher.is_alive()
            assert factory.call_count == 2
        assert first_sock.fileno() == -1 and second_sock.fileno() == -1
        assert any(e['type'] == 'watcher_reconnect' for e in events)
    finally:
        watcher.stop()
        watcher.join(2)
        for client in clients:
            client.close()
        second_peer.close()


def test_watcher_none_rebuilds_are_bounded_and_backoff_is_stoppable():
    hub = MetricsHub()
    events = capture(hub)
    watcher = EncodedWatcher('sub', hub)
    client = MagicMock()
    client.get_frame.return_value = None
    with patch('neoruntime_ipc_sdk.EncodedStreamClient', return_value=client), \
         patch('statusline.WATCHER_REBUILD_LIMIT', 3, create=True), \
         patch('statusline.WATCHER_REBUILD_BACKOFF_S', .01, create=True):
        watcher.start()
        watcher.join(.5)
        stopped_itself = not watcher.is_alive()
        watcher.stop()
        watcher.join(1)
    assert stopped_itself, 'consecutive empty connections must have a rebuild limit'
    assert client.get_frame.call_count == client.close.call_count == 3
    assert watcher.terminal_error == 'reconnect_limit'
    assert events[-1]['type'] == 'watcher_exit'


def test_stop_interrupts_rebuild_backoff_after_socket_is_closed():
    hub = MetricsHub()
    reconnect = threading.Event()
    watcher = EncodedWatcher('sub', hub)
    hub.emit = lambda kind, **fields: reconnect.set() if kind == 'watcher_reconnect' else None
    client = MagicMock()
    client.get_frame.return_value = None
    with patch('neoruntime_ipc_sdk.EncodedStreamClient', return_value=client), \
         patch('statusline.WATCHER_REBUILD_BACKOFF_S', 30, create=True):
        watcher.start()
        reached_backoff = reconnect.wait(.5)
        assert watcher.close(timeout=.5)
    assert reached_backoff
    client.close.assert_called_once()
    client.get_frame.assert_called_once()


@pytest.mark.parametrize('failures', [2, 10])
def test_a_real_sdk_iterator_skips_failed_frames_and_failure_count_is_unknown(failures):
    responses = []
    for seq in range(failures):
        response = inference_pb2.StreamInferResponse(frame_sequence=seq + 1)
        response.status.success = False
        response.status.message = 'test failure'
        responses.append(response)
    if failures < 10:
        response = inference_pb2.StreamInferResponse(frame_sequence=42, timestamp_ns=123)
        response.status.success = True
        responses.append(response)
    async def stream(_request):
        for response in responses:
            yield response
    client = InferenceClient()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    client._loop = loop
    client.stub = NS(StreamInfer=stream)
    hub = MetricsHub()
    events = []
    with patch('chains.a_subscribe.OverlayClient'):
        a = ChainA(camera=MagicMock(), infer=client, hub=hub,
                   metrics_overlay=False, detections_overlay=False)
    def emit(kind, **fields):
        events.append(dict(type=kind, **fields))
        if kind == 'a_error':
            a.stop()
    hub.emit = emit
    a._maybe_annotate = lambda *unused: a.stop()
    try:
        a.start()
        a.join(2)
        assert not a.is_alive()
        snap = hub.a.snapshot()
        assert snap['infer_failures'] is None and snap['infer_attempts'] is None
        assert snap['per_frame_failures_observable'] is False
        assert snap['results'] == (1 if failures < 10 else 0)
        assert snap['subscribe_errors'] == (0 if failures < 10 else 1)
        yielded = [e for e in events if e['type'] == 'a_result']
        assert [e['source_frame_id'] for e in yielded] == ([42] if failures < 10 else [])
    finally:
        a.stop()
        a.join(1)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(1)
        loop.close()
        client._loop = None


def test_a_timestamp_unknown_and_unverified_sdk_latency_not_used_as_source_age():
    hub = MetricsHub()
    events = capture(hub)
    with patch('chains.a_subscribe.OverlayClient'):
        a = ChainA(camera=MagicMock(), infer=MagicMock(), hub=hub,
                   metrics_overlay=False, detections_overlay=False)
    gen = MagicMock()
    gen.__iter__.return_value = iter([(1, NS(timestamp_ns=123, objects=[]))])
    gen.last_latency_ms = gen.avg_latency_ms = 42.0
    gen.last_skew_us = gen.avg_skew_us = gen.dropped = 0
    a.infer.subscribe.return_value = gen
    a._subscribe_loop()
    result = next(e for e in events if e['type'] == 'a_result')
    assert result['result_timestamp_ns'] == 123
    assert result['result_timestamp_clock'] == 'unknown'
    assert 'source_timestamp_ns' not in result
    assert result['latency_ms'] is None and result['source_age_ms'] is None
    assert result['sdk_latency_ms'] == 42.0
    assert hub.a.snapshot()['latency']['n'] == 0


def test_run_start_declares_separate_timestamp_domains():
    app = PerfDemoApp(app_args('--chains', 'none', '--json-path', '', '--watch-encoded', ''))
    events = capture(app.hub)
    with patch('neoruntime_ipc_sdk.CameraClient', side_effect=OSError('offline')):
        assert app.run() == 1
    start = events[0]
    assert 'source_clock' not in start
    assert start['clock_domains']['a_result_timestamp_ns'] == 'unknown'
    assert start['clock_domains']['b_source_timestamp_ns'] == 'device_CLOCK_MONOTONIC'
    b = ChainB(camera=MagicMock(), infer=MagicMock(), dsp=MagicMock(), media=MagicMock(), hub=MetricsHub())
    source = b._source(NS(sequence=1, timestamp_ns=123))
    assert source['source_timestamp_clock'] == 'device_CLOCK_MONOTONIC'
    assert 'result_timestamp_ns' not in source


@pytest.mark.parametrize('case,kept,reason', [
    ('normal', False, 'unregistered'),
    ('keep', True, 'keep_requested'),
    ('reused', True, 'reused_registration'),
    ('b_error', True, 'cleanup_skipped_b_error'),
    ('active', True, 'cleanup_skipped_active_workers'),
    ('unregister_error', None, 'unregister_outcome_unknown'),
    ('register_error', None, 'registration_outcome_unknown'),
    ('missing_reuse', False, 'not_registered'),
    ('not_inspected', None, 'not_inspected'),
])
def test_model_kept_tracks_actual_registration_cleanup_outcome(case, kept, reason):
    extra = ['--keep-model'] if case == 'keep' else ['--reuse-model'] if case in ('reused', 'missing_reuse') else []
    app = PerfDemoApp(app_args(*extra))
    events = capture(app.hub)
    infer = MagicMock()
    infer.get_model_info.return_value = NS(model_path='/models/test.hef') if case == 'reused' else None
    if case == 'register_error':
        infer.register_model.side_effect = OSError('unknown RPC outcome')
    if case not in ('not_inspected',):
        if case in ('register_error', 'missing_reuse'):
            with pytest.raises((OSError, ValueError)):
                app._prepare_model(infer)
        else:
            app._prepare_model(infer)
    if case == 'b_error':
        app.hub.b.counters.set('cleanup_error', 'pacer join timeout')
    if case == 'active':
        worker = MagicMock()
        worker.name, worker.is_alive.return_value = 'blocked', True
        app._threads = [worker]
    if case == 'unregister_error':
        infer.unregister_model.side_effect = OSError('unknown RPC outcome')
    app._teardown(infer, None, 0)
    final = next(e for e in events if e['type'] == 'run_exit')
    assert final['model_kept'] is kept
    assert final['model_kept_reason'] == reason
    if case not in ('normal', 'unregister_error'):
        infer.unregister_model.assert_not_called()


def aio_error(status_code):
    return grpc.aio.AioRpcError(
        status_code, grpc.aio.Metadata(), grpc.aio.Metadata(),
        details='Model not found', debug_error_string='test lookup failure')


def test_lookup_real_aio_not_found_registers_new_test_model():
    app = PerfDemoApp(app_args('--keep-model'))
    infer = MagicMock()
    infer.get_model_info.side_effect = aio_error(grpc.StatusCode.NOT_FOUND)
    app._prepare_model(infer)
    infer.register_model.assert_called_once_with(
        '/models/test.hef', model_id='test-owned', owner_id='perf-demo', model_type='detection',
        model_variant=None)
    assert app._model_state['model_ownership'] == 'created_by_run'
    app._cleanup_model(infer)
    infer.unregister_model.assert_not_called()


def test_variant_flag_passthrough_to_register_model():
    app = PerfDemoApp(app_args('--variant', 'hailo_yolov8n'))
    infer = MagicMock()
    infer.get_model_info.side_effect = aio_error(grpc.StatusCode.NOT_FOUND)
    app._prepare_model(infer)
    infer.register_model.assert_called_once_with(
        '/models/test.hef', model_id='test-owned', owner_id='perf-demo', model_type='detection',
        model_variant='hailo_yolov8n')


def test_lookup_real_aio_not_found_still_rejects_reuse():
    app = PerfDemoApp(app_args('--reuse-model'))
    infer = MagicMock()
    infer.get_model_info.side_effect = aio_error(grpc.StatusCode.NOT_FOUND)
    with pytest.raises(ValueError, match='reuse requested but model ID does not exist'):
        app._prepare_model(infer)
    infer.register_model.assert_not_called()
    infer.unregister_model.assert_not_called()
    assert app._model_state['model_kept'] is False


@pytest.mark.parametrize('status_code', [code for code in grpc.StatusCode if code != grpc.StatusCode.NOT_FOUND])
def test_lookup_other_real_aio_status_propagates_without_registering(status_code):
    app = PerfDemoApp(app_args())
    infer = MagicMock()
    error = aio_error(status_code)  # identical details must not cause a text-based match
    infer.get_model_info.side_effect = error
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        app._prepare_model(infer)
    assert caught.value is error
    infer.register_model.assert_not_called()
    assert app._model_state['model_kept'] is None


def test_lookup_non_rpc_not_found_text_is_not_treated_as_absent():
    app = PerfDemoApp(app_args())
    infer = MagicMock()
    error = RuntimeError('Model not found')
    infer.get_model_info.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        app._prepare_model(infer)
    assert caught.value is error
    infer.register_model.assert_not_called()
