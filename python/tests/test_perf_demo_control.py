"""Host-only capability/regression evals for demo measurement controls."""
import copy
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from neoruntime_ipc_sdk.draw import render_overlay_fragments, render_overlay_rgba
from neoruntime_ipc_sdk.inference_types import BoundingBox, DetectedObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/perf_demo'))
from app import PerfDemoApp, parse_args
from chains.a_subscribe import ChainA
from chains.b_keepfd import ChainB, COMPOSE_FAIL_ESCALATE
from metrics import MetricsHub
from statusline import StatusLine


def args(*extra):
    return parse_args(['--model-path', '/models/test.hef', *extra])


def test_control_defaults_and_none_baseline():
    assert args().chains == 'ab'
    assert args().b_fps == 0
    assert args().publish_hz == 40
    a = args('--chains', 'none', '--b-fps', '10', '--model-id', 'test-only',
             '--reuse-model', '--keep-model', '--no-metrics-overlay',
             '--no-detections-overlay', '--run-id', 'run', '--phase', 'warmup')
    assert a.chains == 'none' and a.b_fps == 10 and a.reuse_model


@pytest.mark.parametrize('flag,value', [
    ('--b-fps', '-1'), ('--b-fps', 'nan'), ('--b-fps', 'inf'),
    ('--b-fps', '121'), ('--publish-hz', '0'), ('--a-fps', '0'),
    ('--annotate-hz', '0'), ('--duration', '-1'), ('--min-score', '1.1'),
    ('--pool-depth', '9'), ('--b-skip', '-1'), ('--status-interval', '0'),
    ('--model-id', ''), ('--phase', ''), ('--run-id', ''),
    ('--variant', 'bad name'), ('--variant', 'x' * 129), ('--variant', ''),
])
def test_invalid_controls_fail_before_clients(flag, value):
    with pytest.raises(SystemExit):
        args(flag, value)


def test_thread_stop_does_not_shadow_thread_internals():
    hub = MetricsHub()
    with patch('chains.a_subscribe.OverlayClient'):
        a = ChainA(camera=MagicMock(), infer=MagicMock(), hub=hub)
    b = ChainB(camera=MagicMock(), infer=MagicMock(), dsp=MagicMock(),
               media=MagicMock(), hub=hub)
    s = StatusLine(hub=hub, camera=MagicMock(), infer=MagicMock(),
                   model_id='test', json_path='')
    s.tick = MagicMock()
    for t in (a, b, s):
        t.stop()
        t.start()
        t.join(1)
        assert not t.is_alive()


def make_b(frames, **kwargs):
    hub = MetricsHub()
    events = []
    hub.emit = lambda kind, **fields: events.append(dict(type=kind, **fields))
    media = MagicMock()
    b = ChainB(camera=MagicMock(), infer=MagicMock(), dsp=MagicMock(),
               media=media, hub=hub, **kwargs)
    pending = iter(frames)
    def receive(*a, **kw):
        f = next(pending, None)
        if f is None:
            b.stop()
        return f
    media.get_frame.side_effect = receive
    b._infer_wh = (640, 384)
    b._overlays_for = MagicMock(return_value=[])
    b._publisher = MagicMock()
    return b, events


def frame(seq=1):
    f = MagicMock()
    f.sequence = seq
    f.timestamp_ns = time.monotonic_ns() - 10_000_000
    f.__enter__.return_value = f
    return f


@pytest.mark.parametrize('result,success', [
    (None, 0), (NS(objects=[], infer_time_us=900, hw_infer_time_us=0,
                  queue_time_us=7), 1),
])
def test_frame_completion_is_not_inference_success(result, success):
    f = frame()
    b, events = make_b([f])
    b.infer.infer.return_value = result
    b._frame_loop()
    snap = b.hub.b.snapshot()
    assert snap['frames_ok'] == 1
    assert snap['infer_attempts'] == 1
    assert snap['infer_success'] == success
    assert snap['infer_failures'] == 1 - success
    assert snap['hw_infer']['n'] == 0
    assert snap['compose_age'] == snap['e2e']
    sample = next(e for e in events if e['type'] == 'b_compose')
    assert sample['source_frame_id'] == 1
    assert sample['hw_infer_time_us'] is None
    assert sample['compose_done_ns'] >= sample['source_timestamp_ns']
    f.__exit__.assert_called_once()


def test_rate_limit_releases_skipped_frames_without_materializing():
    frames = [frame(i) for i in range(3)]
    b, events = make_b(frames, fps=1)
    b._frame_loop()
    assert b.hub.b.snapshot()['rate_skipped'] == 2
    assert b.hub.b.snapshot()['frames_delivered'] == 3
    for f in frames[1:]:
        f.__exit__.assert_called_once()
        f.to_array.assert_not_called()
    assert sum(e['type'] == 'b_delivery' for e in events) == 3


def test_recorder_appends_and_exposes_final_loss(tmp_path):
    from recorder import SampleRecorder
    path = tmp_path / 'samples.jsonl'
    path.write_text('{"old":true}\n')
    r = SampleRecorder(str(path), run_id='run', phase='测量', capacity=8)
    for i in range(10000):
        r.emit('sample', source_frame_id=i, value='unicode;\"')
    assert r.close(timeout=3)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0] == {'old': True}
    final = rows[-1]
    assert final['type'] == 'recorder_final'
    assert final['written'] + final['dropped'] == 10000
    assert final['dropped'] == r.snapshot()['dropped']
    assert all(e['run_id'] == 'run' and e['phase'] == '测量' and
               isinstance(e['monotonic_ns'], int) for e in rows[1:])


def test_reuse_model_never_registers_or_unregisters_existing():
    app = PerfDemoApp(args('--model-id', 'test-only', '--reuse-model'))
    infer = MagicMock()
    infer.get_model_info.return_value = NS(model_path='/models/test.hef',
                                           load_timestamp=123)
    app._prepare_model(infer)
    app._cleanup_model(infer)
    infer.register_model.assert_not_called()
    infer.unregister_model.assert_not_called()


def test_existing_model_rejected_without_reuse():
    app = PerfDemoApp(args('--model-id', 'test-only'))
    infer = MagicMock()
    infer.get_model_info.return_value = NS(model_path='/models/other.hef')
    with pytest.raises(ValueError):
        app._prepare_model(infer)
    infer.register_model.assert_not_called()


@pytest.mark.parametrize('selected,expected', [('none', []), ('a', ['a']), ('b', ['b']), ('ab', ['b', 'a'])])
def test_app_mocked_end_to_end_selection_cleanup_and_jsonl(tmp_path, selected, expected):
    path = tmp_path / 'events.jsonl'
    app = PerfDemoApp(args('--chains', selected, '--samples-path', str(path),
                           '--watch-encoded', '', '--json-path', '', '--model-id', 'only-test'))
    clients = {name: MagicMock() for name in ('CameraClient', 'InferenceClient', 'DspClient', 'FdMediaClient')}
    clients['InferenceClient'].get_model_info.return_value = None
    made = []
    def worker(label):
        def create(**kwargs):
            t = MagicMock()
            t.name = label
            t.is_alive.return_value = False
            made.append(label)
            return t
        return create
    with patch.multiple('neoruntime_ipc_sdk', **{k: MagicMock(return_value=v) for k, v in clients.items()}), \
         patch('app.ChainA', side_effect=worker('a')), patch('app.ChainB', side_effect=worker('b')), \
         patch('app.StatusLine', side_effect=worker('status')), patch.object(app, '_watchdog'):
        assert app.run() == 0
    assert made == expected + ['status']
    clients['InferenceClient'].unregister_model.assert_called_once_with('only-test')
    clients['InferenceClient'].close.assert_called_once()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['type'] == 'run_start'
    assert rows[-2]['type'] == 'run_exit' and rows[-2]['exit_code'] == 0
    assert rows[-1]['type'] == 'recorder_final'


@pytest.mark.parametrize('keep', [False, True])
def test_new_model_cleanup_only_owns_new_registration(keep):
    app = PerfDemoApp(args(*(['--keep-model'] if keep else [])))
    infer = MagicMock()
    infer.get_model_info.return_value = None
    app._prepare_model(infer)
    app._cleanup_model(infer)
    assert infer.unregister_model.call_count == (0 if keep else 1)
    infer.register_model.assert_called_once()


def test_missing_reuse_and_mismatch_reject_without_mutation():
    app = PerfDemoApp(args('--model-id', 'only-test', '--reuse-model'))
    infer = MagicMock()
    for info in (None, NS(model_path='/other.hef')):
        infer.get_model_info.return_value = info
        with pytest.raises(ValueError):
            app._prepare_model(infer)
    infer.register_model.assert_not_called()


@pytest.mark.parametrize('extra', [('--reuse-model',), ('--samples-path', '/same', '--json-path', '/same')])
def test_unsafe_cli_combinations_rejected(extra):
    with pytest.raises(SystemExit):
        args(*extra)


def test_watcher_consumes_packets_and_closes_worker_owned_client():
    from statusline import EncodedWatcher
    hub = MetricsHub()
    events = []
    hub.emit = lambda kind, **fields: events.append(dict(type=kind, **fields))
    watcher = EncodedWatcher('sub', hub)
    client = MagicMock()
    pending = iter([NS(pts_ns=10_000_000, seq=1), NS(pts_ns=20_000_000, seq=2)])
    def get(**kw):
        item = next(pending, 'done')
        if item == 'done':
            watcher.stop()
            return None
        return item
    client.get_frame.side_effect = get
    with patch('neoruntime_ipc_sdk.EncodedStreamClient', return_value=client):
        watcher.start()
        watcher.join(1)
        assert watcher.close()
    client.close.assert_called_once()
    assert hub.snapshot()['sys']['encode_gap']['sub']['mean'] == 10
    assert [e['packet_sequence'] for e in events if e['type'] == 'encoded_packet'] == [1, 2]


def test_watcher_network_error_and_close_error_are_visible():
    from statusline import EncodedWatcher
    hub = MetricsHub()
    watcher = EncodedWatcher('sub', hub)
    client = MagicMock()
    def fail(**kw):
        watcher.stop()
        raise OSError('offline')
    client.get_frame.side_effect = fail
    client.close.side_effect = OSError('close')
    with patch('neoruntime_ipc_sdk.EncodedStreamClient', return_value=client):
        watcher.start()
        watcher.join(1)
    assert watcher.cleanup_error == 'OSError'


def test_pacer_records_unique_source_and_repeats_and_stops():
    b, events = make_b([])
    source = dict(source_frame_id=3, source_timestamp_ns=123, compose_done_ns=456, stream_id='sub')
    b._latest = ('pixels', source)
    def publish(*a, **kw):
        if b._publisher.publish.call_count == 3:
            b.stop()
    b._publisher.publish.side_effect = publish
    b._pacer_loop()
    snap = b.hub.b.snapshot()
    assert snap['publish_calls'] == snap['publishes'] == 3
    assert snap['published_unique'] == 1
    publishes = [e for e in events if e['type'] == 'b_publish']
    assert all(e['source_frame_id'] == 3 and e['compose_done_ns'] == 456 for e in publishes)
    assert all(e['publish_ms'] >= 0 for e in publishes)


def test_publish_failures_and_close_failures_visible():
    b, events = make_b([])
    pub = b._publisher
    pub.publish.side_effect = OSError('offline')
    source = dict(source_frame_id=1, source_timestamp_ns=1, compose_done_ns=2)
    for _ in range(10):
        b._publish(pub, 'pixels', source, 3)
    assert b.hub.b.snapshot()['publish_errors'] == 10
    assert b.hub.b.snapshot()['degraded_reason']
    pub.publish_eos.side_effect = OSError('eos')
    pub.close.side_effect = OSError('close')
    assert b._close_publisher(eos=True)
    assert b.hub.b.snapshot()['cleanup_error'] == 'OSError'
    assert all(not e['success'] for e in events)


@pytest.mark.parametrize('stage', ['resize', 'rpc', 'materialize', 'blend'])
def test_pipeline_errors_release_frame_and_keep_failure_stage(stage):
    f = frame()
    b, events = make_b([f])
    calls = {'resize': b.dsp.resize_hw, 'rpc': b.infer.infer,
             'materialize': f.to_array, 'blend': b.dsp.blend_hw}
    calls[stage].side_effect = OSError('offline')
    # Compose stages (materialize/blend) are absorbed by the self-heal
    # contract: one transient failure counts and backs off, no escalation.
    b._frame_loop()
    f.__exit__.assert_called_once()
    if stage in ('resize', 'rpc'):
        sample = next(e for e in events if e['type'] == 'b_infer')
        assert sample['failure_stage'] == stage and not sample['success']
        assert b.hub.b.snapshot()['infer_attempts'] == int(stage == 'rpc')
    else:
        assert b.hub.b.snapshot()['compose_failures_transient'] == 1
        assert not [e for e in events if e['type'] == 'b_compose' and e['success']]


def test_compose_failures_escalate_after_consecutive_streak():
    frames = [frame(seq=i + 1) for i in range(COMPOSE_FAIL_ESCALATE)]
    b, events = make_b(frames)
    b.dsp.blend_hw.side_effect = OSError('offline')
    b._stop_event.wait = MagicMock()  # skip the real backoff sleeps
    with pytest.raises(OSError):
        b._frame_loop()
    snap = b.hub.b.snapshot()
    assert snap['compose_failures_transient'] == COMPOSE_FAIL_ESCALATE
    b._stop_event.wait.assert_called()  # backoff was engaged before escalation


def test_zero_copy_hw_timing_and_no_overlays():
    f = frame()
    b, events = make_b([f], zero_copy=True, metrics_overlay=False, detections_overlay=False)
    del b._overlays_for  # exercise the real empty-overlay path
    b.infer.infer.return_value = NS(objects=[], infer_time_us=900, hw_infer_time_us=123, queue_time_us=7)
    b._frame_loop()
    f.to_array.assert_not_called()
    assert b.hub.b.snapshot()['hw_infer']['mean'] == .123
    assert b.hub.b.snapshot()['publishes'] == 1
    infer_event = next(e for e in events if e['type'] == 'b_infer')
    assert infer_event['daemon_infer_time_us'] == 900
    assert infer_event['queue_time_us'] == 7
    assert infer_event['hw_infer_time_us'] == 123


def test_model_geometry_and_overlay_rendering():
    b, _ = make_b([])
    for shape in ([1, 384, 640, 3], [384, 640, 3]):
        b.infer.get_model_info.return_value = NS(inputs=[{'shape': shape}])
        b._prepare_infer()
        assert b._infer_wh == (640, 384)
    for shape in ([], [1], [0, 640, 3]):
        b.infer.get_model_info.return_value = NS(inputs=[NS(shape=shape)])
        with pytest.raises(ValueError):
            b._prepare_infer()
    del b._overlays_for
    f = NS(width=640, height=384)
    out = NS(objects=[DetectedObject('test', .8, BoundingBox(.25, .25, .5, .5))])
    with patch('chains.b_keepfd.render_overlay_fragments',
               return_value=[('rgba', 0, 0)]):
        assert len(b._overlays_for(f, out)) == 2
        assert b._chip_for(640, 384) is b._chip_for(640, 384)


@pytest.mark.parametrize('width,height', [(1280, 720), (720, 1280), (640, 384)])
def test_b_normalized_boxes_use_real_draw_source_pixel_contract(width, height):
    b, _ = make_b([], metrics_overlay=False)
    del b._overlays_for
    obj = DetectedObject('person', .8, BoundingBox(.625, .25, .25, .5))
    out = NS(objects=[obj])
    original = copy.deepcopy(out)
    expected_box = (.625 * width, .25 * height, .875 * width, .75 * height)
    expected = render_overlay_fragments(
        width, height, boxes=[expected_box], labels=['person'], scores=[.8])

    # Execute the actual SDK renderer, not a stub returning an arbitrary canvas.
    with patch('chains.b_keepfd.render_overlay_fragments',
               wraps=render_overlay_fragments) as draw:
        for _ in range(2):
            frags = b._overlays_for(NS(width=width, height=height), out)
            assert len(frags) == len(expected)
            for (rgba, x0, y0), (exp, ex0, ey0) in zip(frags, expected):
                assert (x0, y0) == (ex0, ey0)
                np.testing.assert_array_equal(rgba, exp)
            # A real opaque stroke must land on the right-side source-frame
            # edge: reassemble the fragments into a frame-space alpha map.
            alpha = np.zeros((height, width), np.uint8)
            for rgba, x0, y0 in frags:
                h, w = rgba.shape[:2]
                np.maximum(alpha[y0:y0 + h, x0:x0 + w], rgba[..., 3],
                           out=alpha[y0:y0 + h, x0:x0 + w])
            assert alpha[int(.5 * height), int(.875 * width) - 1] == 255
            assert draw.call_args.kwargs['boxes'] == [expected_box]
            assert draw.call_args.kwargs['labels'] == ['person']
            assert draw.call_args.kwargs['scores'] == [.8]
            assert out == original
            assert out.objects[0] is obj
        assert draw.call_count == 2


def test_b_normalized_boxes_rescale_once_per_source_without_mutating_results():
    b, _ = make_b([], metrics_overlay=False, min_score=.3)
    del b._overlays_for
    labels = ['', "目标 ' ; --", 'edge']
    out = NS(objects=[
        DetectedObject(labels[0], .3, BoundingBox(.25, .25, .5, .5)),
        DetectedObject(labels[1], 1., BoundingBox(0., 0., 1., 1.)),
        DetectedObject(labels[2], .9, BoundingBox(.5, .5, 0., 0.)),
        DetectedObject('filtered', .299, BoundingBox(.1, .1, .1, .1)),
    ])
    original = copy.deepcopy(out)
    with patch('chains.b_keepfd.render_overlay_fragments',
               wraps=render_overlay_fragments) as draw:
        for width, height in [(1280, 720), (720, 1280), (1280, 720)]:
            b._overlays_for(NS(width=width, height=height), out)
            # Tight stroke fragments per box (not the union bbox): a
            # spread-out scene must not pay frame-sized render+blend for
            # untouched pixels.
            calls = draw.call_args_list[-3:]
            assert [c.kwargs['boxes'] for c in calls] == [
                [(.25 * width, .25 * height, .75 * width, .75 * height)],
                [(0., 0., float(width), float(height))],
                [(.5 * width, .5 * height, .5 * width, .5 * height)],
            ]
            assert [c.kwargs['labels'] for c in calls] == [[l] for l in labels]
            assert [c.kwargs['scores'] for c in calls] == [[.3], [1.], [.9]]
            assert out == original


@pytest.mark.parametrize('out,enabled', [
    (None, True), (NS(objects=None), True), (NS(objects=[]), True),
    (NS(objects=[DetectedObject('low', .1, BoundingBox(0., 0., 1., 1.))]), True),
    (NS(objects=[DetectedObject('off', .9, BoundingBox(0., 0., 1., 1.))]), False),
])
def test_b_normalized_boxes_empty_or_disabled_do_not_render(out, enabled):
    b, _ = make_b([], metrics_overlay=False, detections_overlay=enabled)
    del b._overlays_for
    with patch('chains.b_keepfd.render_overlay_fragments') as draw:
        assert b._overlays_for(NS(width=1280, height=720), out) == []
        draw.assert_not_called()


def test_b_normalized_boxes_pipeline_blends_source_pixels_not_model_pixels():
    f = frame()
    f.width, f.height = 1280, 720
    b, events = make_b([f], metrics_overlay=False)
    del b._overlays_for
    out = NS(objects=[DetectedObject('person', .8, BoundingBox(.625, .25, .25, .5))])
    original = copy.deepcopy(out)
    b.infer.infer.return_value = out
    expected = render_overlay_fragments(
        1280, 720, boxes=[(800, 180, 1120, 540)], labels=['person'], scores=[.8])

    b._frame_loop()

    b.dsp.resize_hw.assert_called_once_with(f, 640, 384, scaling='stretch')
    b.infer.infer.assert_called_once_with(b.dsp.resize_hw.return_value, b.model_id)
    b.dsp.blend_hw.assert_called_once()
    args, kwargs = b.dsp.blend_hw.call_args
    assert args[0] is f.to_array.return_value
    assert len(args[1]) == len(expected)
    for (rgba, x0, y0), (exp, ex0, ey0) in zip(args[1], expected):
        assert (x0, y0) == (ex0, ey0)
        np.testing.assert_array_equal(rgba, exp)
    assert kwargs == {'zero_copy': False}
    assert b._latest[0] is b.dsp.blend_hw.return_value
    assert next(e for e in events if e['type'] == 'b_compose')['success'] is True
    assert out == original
    f.__exit__.assert_called_once()


@pytest.mark.parametrize('reported', [True, False, None, 1, 'unknown', 'missing'])
def test_b_compose_backend_snapshots_only_successful_blend_report(reported):
    f = frame()
    b, events = make_b([f])
    b.dsp.resize_hw.side_effect = lambda *a, **kw: setattr(b.dsp, 'last_used_hw', True)
    def blend(*a, **kw):
        if reported == 'missing':
            del b.dsp.last_used_hw
        else:
            b.dsp.last_used_hw = reported
        return 'composed'
    b.dsp.blend_hw.side_effect = blend
    b._compose_frame(f, b._source(f), time.monotonic_ns())
    sample = next(e for e in events if e['type'] == 'b_compose')
    assert sample['success'] is True
    assert sample['dsp_backend'] is (reported if type(reported) is bool else None)
    assert sample['overlay_geometry_scope'] == 'app_blend_input'
    assert sample['overlay_geometry'] == []


@pytest.mark.parametrize('stage', ['materialize', 'render', 'blend'])
def test_b_compose_failure_backend_null_and_retains_available_geometry(stage):
    f = frame()
    b, events = make_b([])
    rgba = np.zeros((18, 22, 4), dtype=np.uint8)
    b._overlays_for.return_value = [(rgba, 2, 4)]
    b.dsp.last_used_hw = True
    b._compose_frame(f, b._source(f), time.monotonic_ns())
    assert events[-1]['dsp_backend'] is True
    operation = {'materialize': f.to_array, 'render': b._overlays_for,
                 'blend': b.dsp.blend_hw}[stage]
    error = OSError('failed operation')
    operation.side_effect = error
    with pytest.raises(OSError) as caught:
        b._compose_frame(f, b._source(f), time.monotonic_ns())
    assert caught.value is error
    sample = events[-1]
    assert sample['success'] is False and sample['dsp_backend'] is None
    assert sample['error'] == 'OSError'
    expected = [dict(x=2, y=4, w=22, h=18, stride=88, bytes=1584)]
    assert sample['overlay_geometry'] == (expected if stage == 'blend' else None)


@pytest.mark.parametrize('broken_metadata', [False, True])
@pytest.mark.parametrize('blend_fails', [False, True])
def test_b_compose_geometry_reports_actual_input_without_affecting_result(broken_metadata, blend_fails):
    f = frame()
    b, events = make_b([])
    rgba = np.zeros((24, 60, 4), dtype=np.uint8)[:, ::2, :]
    chip = np.zeros((16, 20, 4), dtype=np.uint8)
    original = rgba.copy()
    overlays = [(rgba, np.int64(7), np.int64(9)), (chip, 4, 100)]
    if broken_metadata:
        overlays.append((NS(shape=(16, 20, 4), strides=(object(),), nbytes=1280), 0, 0))
    b._overlays_for.return_value = overlays
    b.infer.infer.return_value = NS(objects=[])
    b.dsp.last_used_hw = False
    if blend_fails:
        b.dsp.blend_hw.side_effect = OSError('blend failed')
        with pytest.raises(OSError, match='blend failed'):
            b._compose_frame(f, b._source(f), time.monotonic_ns())
    else:
        b._compose_frame(f, b._source(f), time.monotonic_ns())
        assert b._latest[0] is b.dsp.blend_hw.return_value
    sample = events[-1]
    assert sample['success'] is (not blend_fails)
    assert sample['dsp_backend'] is (None if blend_fails else False)
    expected = [dict(x=7, y=9, w=30, h=24, stride=240, bytes=2880),
                dict(x=4, y=100, w=20, h=16, stride=80, bytes=1280)]
    assert sample['overlay_geometry'] == (None if broken_metadata else expected)
    json.dumps(sample, allow_nan=False)
    assert b.dsp.blend_hw.call_args.args[1] is overlays
    np.testing.assert_array_equal(rgba, original)


@pytest.mark.parametrize('diagnostic_raises', [False, True])
def test_b_compose_slow_backend_diagnostic_does_not_pollute_timing(diagnostic_raises):
    f = frame()
    f.timestamp_ns = start = 1_000_000_000
    b, events = make_b([])
    b.infer.infer.return_value = NS(objects=[])
    clock = [start]
    def blend(*args, **kwargs):
        clock[0] += 10_000_000
        return 'composed'
    def diagnostic():
        clock[0] += 25_000_000
        if diagnostic_raises:
            raise RuntimeError('unreadable diagnostic')
        return False
    b.dsp.blend_hw.side_effect = blend
    with patch('chains.b_keepfd.time.monotonic_ns', side_effect=lambda: clock[0]), \
         patch.object(type(b.dsp), 'last_used_hw', new_callable=PropertyMock, create=True) as flag:
        flag.side_effect = diagnostic
        b._compose_frame(f, b._source(f), start)
    sample = events[-1]
    assert sample['success'] is True and sample['error'] is None
    assert sample['dsp_backend'] is (None if diagnostic_raises else False)
    assert sample['blend_ms'] == sample['compose_ms'] == sample['source_age_ms'] == 10.0
    assert sample['render_ms'] == 0.0
    assert sample['compose_done_ns'] == b._latest[1]['compose_done_ns'] == start + 10_000_000
    assert clock[0] == start + 35_000_000
    flag.assert_called_once_with()


def test_a_results_annotation_failures_and_iterator_cleanup():
    hub = MetricsHub()
    events = []
    hub.emit = lambda kind, **fields: events.append(dict(type=kind, **fields))
    with patch('chains.a_subscribe.OverlayClient'):
        a = ChainA(camera=MagicMock(), infer=MagicMock(), hub=hub)
    a.camera.get_stream_status.return_value = [NS(stream_id='main', stream_epoch=2)]
    result = NS(objects=[NS(score=.9)], timestamp_ns=1)
    gen = MagicMock()
    gen.__iter__.return_value = iter([(4, result)])  # SDK never yields failed/None results
    gen.last_latency_ms, gen.last_skew_us = 2, 3
    gen.avg_latency_ms, gen.avg_skew_us = 2, 3
    gen.dropped = 0
    a.infer.subscribe.return_value = gen
    a._subscribe_loop()
    gen.cancel.assert_called_once()
    gen.close.assert_called_once()
    assert hub.a.snapshot()['results'] == 1
    assert len([e for e in events if e['type'] == 'a_annotate']) == 2
    a._maybe_annotate(4, result, a._last_annotate)
    assert hub.a.snapshot()['annotate_calls'] == 2
    a.overlay.annotate.side_effect = OSError('offline')
    a._maybe_annotate(4, result, a._last_annotate + 1)
    assert hub.a.snapshot()['annotate_errors'] == 1
    a.degrade('test stop')
    assert hub.a.snapshot()['degraded_reason'] == 'test stop'


def test_a_no_overlays_and_subscription_error_exit():
    hub = MetricsHub()
    with patch('chains.a_subscribe.OverlayClient'):
        a = ChainA(camera=MagicMock(), infer=MagicMock(), hub=hub,
                   metrics_overlay=False, detections_overlay=False)
    a._maybe_annotate(0, NS(objects=[]), time.monotonic())
    a.overlay.annotate.assert_not_called()
    def fail():
        a.stop()
        raise OSError('offline')
    a._subscribe_loop = fail
    a.start()
    a.join(1)
    assert not a.is_alive()
    a.overlay.close.assert_called_once()
    assert hub.a.snapshot()['subscribe_errors'] == 1


def test_status_sampling_and_io_failures(tmp_path):
    hub = MetricsHub()
    camera, infer = MagicMock(), MagicMock()
    status = StatusLine(hub=hub, camera=camera, infer=infer, model_id='test',
                        json_path=str(tmp_path / 'status.json'))
    infer.get_stats.return_value = {'device_utilization': 20, 'model_stats': [
        {'model_id': 'other'}, {'model_id': 'test', 'hw_fps': 10}]}
    camera.injection_status.return_value = NS(frames_dropped=3, in_flight_buffer_ids=[1])
    camera.get_stream_status.return_value = [NS(stream_id='main', packets_published=10,
        bake_skips=1, stream_epoch=2, overlay_late_commands=0)]
    status.tick()
    camera.get_stream_status.return_value = [NS(stream_id='main', packets_published=20,
        bake_skips=2, stream_epoch=2, overlay_late_commands=1)]
    status.tick()
    snap = json.loads((tmp_path / 'status.json').read_text())
    assert snap['sys']['stream_delta']['main']['bake_pct'] == 90
    assert snap['sys']['model_hw_fps'] == 10
    infer.get_stats.assert_called_with(sampling_window_ms=50)
    camera.get_stream_status.side_effect = OSError('offline')
    camera.injection_status.side_effect = OSError('offline')
    infer.get_stats.side_effect = OSError('offline')
    status.tick()
    status.json_path = str(tmp_path)  # directory target cannot be replaced
    status._write_json({})


@pytest.mark.parametrize('kwargs', [{'capacity': 0}, {'capacity': None}, {'capacity': True}, {'batch_size': -1}])
def test_recorder_invalid_bounds(kwargs, tmp_path):
    from recorder import SampleRecorder
    with pytest.raises(ValueError):
        SampleRecorder(str(tmp_path / 'samples'), run_id='run', phase='p', **kwargs)


def test_recorder_limit_invalid_json_and_closed_producer(tmp_path):
    from recorder import SampleRecorder
    for fields, limit in [({'value': object()}, 10000), ({'value': 'too big'}, 1)]:
        path = tmp_path / str(limit)
        r = SampleRecorder(str(path), run_id='run', phase='p', max_bytes=limit)
        r.emit('sample', **fields)
        assert r.close(2)
        assert r.snapshot()['dropped'] == 1
        assert r.snapshot()['error'] is not None
        assert not r.emit('sample')
        with pytest.raises(ValueError):
            r.emit(None)
        final = json.loads(path.read_text().splitlines()[-1])
        assert final['error'] is not None


def test_recorder_concurrent_producers_account_for_all_items(tmp_path):
    from recorder import SampleRecorder
    path = tmp_path / 'concurrent'
    r = SampleRecorder(str(path), run_id='r', phase='p', capacity=64)
    workers = [threading.Thread(target=lambda: [r.emit('sample') for _ in range(2500)]) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert r.close(3)
    snap = r.snapshot()
    assert snap['written'] + snap['dropped'] == 10000
    assert snap['pending'] == 0


def test_recorder_terminal_exit_code_reflects_delayed_write_failure(tmp_path):
    from recorder import SampleRecorder
    path = tmp_path / 'limited'
    r = SampleRecorder(str(path), run_id='r', phase='p', max_bytes=1)
    r.emit('sample')
    assert r.close(exit_code=0)
    final = json.loads(path.read_text().splitlines()[-1])
    assert final['exit_code'] == 1


def test_rate_limit_boundary_and_decimation_and_unknown_source_age():
    b, _ = make_b([], fps=10, skip=2)
    b._delivery_count = 1
    assert b._skip_reason(1_000_000_000) is None
    b._delivery_count = 2
    assert b._skip_reason(1_100_000_000) == 'decimation'
    b._delivery_count = 3
    assert b._skip_reason(1_099_999_999) == 'rate_limit'
    assert b._skip_reason(1_100_000_000) is None
    b.stop()
    assert b._skip_reason(2_000_000_000) == 'stopping'
    for source in (None, 0, -1, 101):
        assert b._age_ms(100, source) is None


def test_b_run_error_cleans_media_and_publisher_without_rebuilding_after_stop():
    b, events = make_b([])
    def fail():
        b.stop()
        raise OSError('offline')
    b._prepare_infer = fail
    b.media.close.side_effect = OSError('close')
    b.start()
    b.join(1)
    assert not b.is_alive()
    assert b.hub.b.snapshot()['cleanup_error'] == 'OSError'
    assert any(e['type'] == 'chain_exit' for e in events)


def test_b_actual_worker_and_pacer_join_closes_lease():
    f = frame()
    b, events = make_b([f])
    b.infer.get_model_info.return_value = NS(inputs=[{'shape': [384, 640, 3]}])
    b.infer.infer.return_value = NS(objects=[])
    pub = MagicMock()
    with patch('neoruntime_ipc_sdk.FramePublisher', return_value=pub):
        b.start()
        b.join(1)
    assert not b.is_alive()
    assert b.hub.b.snapshot()['cleanup_error'] is None
    pub.publish_eos.assert_called_once()
    pub.close.assert_called_once()


def test_pacer_timeout_never_closes_in_use_lease():
    b, _ = make_b([])
    pub = b._publisher
    b._pacer = MagicMock()
    b._pacer.is_alive.return_value = True
    assert not b._close_publisher(eos=True)
    pub.close.assert_not_called()
    assert b.hub.b.snapshot()['cleanup_error'] == 'pacer join timeout'


def test_app_teardown_timeout_reports_failure_and_preserves_in_use_model():
    app = PerfDemoApp(args('--model-id', 'test'))
    worker, infer = MagicMock(), MagicMock()
    worker.name = 'blocked'
    worker.is_alive.return_value = True
    app._threads = [worker]
    app._owns_model = True
    assert app._teardown(infer, None, 0) == 1
    infer.unregister_model.assert_not_called()
    assert app.hub.snapshot()['final']['reason'] == 'failed'


def test_app_startup_failure_closes_constructed_clients(tmp_path):
    app = PerfDemoApp(args('--chains', 'none', '--json-path', '', '--watch-encoded', '',
                           '--samples-path', str(tmp_path / 'raw')))
    camera, infer = MagicMock(), MagicMock()
    infer.get_model_info.side_effect = OSError('offline')
    with patch('neoruntime_ipc_sdk.CameraClient', return_value=camera), \
         patch('neoruntime_ipc_sdk.InferenceClient', return_value=infer):
        assert app.run() == 1
    camera.close.assert_called_once()
    infer.close.assert_called_once()


def test_app_watchdog_cancel_degrade_duration_and_signal():
    app = PerfDemoApp(args('--chains', 'a'))
    app._chain_a = MagicMock()
    app._chains_started_at = time.monotonic() - 30
    with patch('app.WATCHDOG_PERIOD_S', 0), patch('app.A_FIRST_RESULT_DEAD_S', 60):
        app._watchdog(1)
    app._chain_a.cancel_iter.assert_called_once()
    app._stop.clear()
    app._chains_started_at = time.monotonic() - 100
    with patch('app.WATCHDOG_PERIOD_S', 0):
        app._watchdog(1)
    app._chain_a.degrade.assert_called_once()
    app._stop.clear()
    app._on_signal(15, None)
    assert app._stop.is_set()


def test_main_missing_path_and_fatal_run_return_nonzero():
    import app as module
    with patch.object(module, 'parse_args', return_value=args()), patch.object(module.os.path, 'isfile', return_value=False):
        assert module.main() == 2
    with patch.object(module, 'parse_args', return_value=args()), patch.object(module.os.path, 'isfile', return_value=True), \
         patch.object(module.PerfDemoApp, 'run', side_effect=RuntimeError('failure')):
        assert module.main() == 1


def test_status_counter_reset_is_not_negative_delta():
    hub = MetricsHub()
    camera = MagicMock()
    status = StatusLine(hub=hub, camera=camera, infer=MagicMock(), model_id='test', json_path='')
    for packets, epoch in ((100, 2), (10, 3)):
        camera.get_stream_status.return_value = [NS(stream_id='main', packets_published=packets,
            bake_skips=0, stream_epoch=epoch, overlay_late_commands=0)]
        status._sample_stream_deltas()
    assert hub.sys.stream_delta['main']['packets'] is None
    assert hub.sys.stream_delta['main']['reset'] is True


def test_system_sampling_failure_is_explicit_not_stale():
    hub = MetricsHub()
    infer = MagicMock()
    status = StatusLine(hub=hub, camera=MagicMock(), infer=infer, model_id='test', json_path='')
    infer.get_stats.return_value = {'device_utilization': 88}
    status._sample_system()
    infer.get_stats.side_effect = OSError('offline')
    status._sample_system()
    assert hub.sys.snapshot()['npu_util'] is None
    assert hub.sys.snapshot()['sample_error'] == 'OSError'


def test_a_default_keeps_original_overlay_enable_disable_lifecycle():
    with patch('chains.a_subscribe.OverlayClient'):
        a = ChainA(camera=MagicMock(), infer=MagicMock(), hub=MetricsHub())
    a._subscribe_loop = a.stop
    a.start()
    a.join(1)
    a.overlay.enable.assert_called_once_with(show_label=True, show_confidence=False, line_thickness=2)
    a.overlay.disable.assert_called_once()
    a.overlay.close.assert_called_once()
