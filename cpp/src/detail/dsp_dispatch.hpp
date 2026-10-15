// dsp_dispatch.hpp — the DSP dispatch surface the accel router's resident
// machinery runs jobs through (python accel.py's module globals
// _dsp_shared / _dsp_active / _dsp_retired + the _lazy_dsp_client
// monkeypatch seam).
//
// DspClient itself is non-virtual, so tests cannot subclass it the way
// python's TestSharedResident patches the factory: this interface is the
// seam instead. DefaultDspDispatch (accel.cpp) wraps a process-resident
// DspClient; tests install fakes via install_dsp_dispatch().
//
// Ownership: test fakes are raw pointers the caller owns — the machinery
// closes them when it drains them but never deletes them. The default
// wrapper's storage is a unique_ptr inside the resident state; the state
// itself is leak-on-exit (a `new`ed struct, never destructed) so the
// resident DspClient never sees static destruction order — the process
// exiting closes its sockets and the daemon reclaims the buffers.
#pragma once

#include <any>
#include <memory>
#include <string>
#include <vector>

namespace neoruntime_ipc_sdk {
namespace detail {

class DspDispatch {
public:
    virtual ~DspDispatch() = default;

    // A real client answers false once every pool it handed out is gone;
    // fakes that model transports only inherit false (python
    // _retired_is_idle: no has_live_pools attribute counts as idle).
    virtual bool has_live_pools() { return false; }
    // Idempotent; must not throw from drain (exceptions swallowed there).
    virtual void close() {}
    // One DspClient method, type-erased. See the args convention in
    // neoruntime_ipc_sdk/accel.hpp's shared_dsp_call note.
    virtual std::any call(const std::string& method, std::vector<std::any> args) = 0;
};

// The current resident (python `accel._dsp_shared`); creates the default
// dispatch on first use. Never returns null.
DspDispatch* lazy_dsp_dispatch();

// Retire the resident after a failure — without closing it beneath
// whatever is still using it. A dispatch that is not the current resident
// (a test fake, or one another thread already replaced) is left alone.
void dsp_drop_shared(DspDispatch* dispatch);

// Close retired dispatches once nothing can still be using them: no
// shared call in flight anywhere and no live pool left.
void drain_retired();

// Test inspection only: the current resident, or null (no lazy creation).
DspDispatch* current_resident();

// Test seam: install `dispatch` as the resident — resetting active and
// retired state (python's _install fixture) — or pass nullptr to restore
// the default factory (also resetting state).
void install_dsp_dispatch(DspDispatch* dispatch);

}  // namespace detail
}  // namespace neoruntime_ipc_sdk
