import ctypes

import torch_dbs


def test_torch_ctypes_options_layout_matches_c_abi():
    assert ctypes.sizeof(torch_dbs.DBSOptionsC) == torch_dbs._DBS_OPTIONS_C_SIZE
    for name, expected_offset in torch_dbs._DBS_OPTIONS_C_OFFSETS.items():
        assert getattr(torch_dbs.DBSOptionsC, name).offset == expected_offset

    opts = torch_dbs.DBSOptions(beam_size=4, max_dense_gradient_elements=123_456_789_012)
    c_opts = opts.as_c()
    assert c_opts._pad0 == 0
    assert c_opts.max_dense_gradient_elements == 123_456_789_012


def test_torch_ctypes_library_cache_reuses_loaded_library(monkeypatch):
    torch_dbs._get_dbs_lib.cache_clear()
    constructed = []

    class FakeDBSLib:
        def __init__(self, path):
            self.path = path
            constructed.append(path)

    monkeypatch.setattr(torch_dbs, "_DBSLib", FakeDBSLib)

    first = torch_dbs._get_dbs_lib("/tmp/libdbs-a.so")
    second = torch_dbs._get_dbs_lib("/tmp/libdbs-a.so")
    third = torch_dbs._get_dbs_lib("/tmp/libdbs-b.so")

    assert first is second
    assert third is not first
    assert constructed == ["/tmp/libdbs-a.so", "/tmp/libdbs-b.so"]
    torch_dbs._get_dbs_lib.cache_clear()


def test_torch_ctypes_state_close_is_idempotent():
    calls = []

    class FakeLib:
        @staticmethod
        def dbs_free_result(result):
            calls.append(("free_result", result))

        @staticmethod
        def dbs_destroy(handle):
            calls.append(("destroy", handle))

    class FakeDBS:
        lib = FakeLib()

    state = torch_dbs._CState(FakeDBS(), 17, 29)
    assert not state.closed

    state.close()
    state.close()

    assert state.closed
    assert calls == [("free_result", 29), ("destroy", 17)]
