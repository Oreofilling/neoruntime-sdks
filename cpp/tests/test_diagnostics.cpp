// test_diagnostics.cpp — diagnostics() snapshot and check() grading
// (port of python tests/test_diagnostics.py). Live-probe success paths
// (model_registered=true / stream_active=true) need a reachable daemon
// and are covered by the on-device smokes; here the live probes take
// real-but-unconnected clients and pin the failure branches.
#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/camera.hpp"
#include "neoruntime_ipc_sdk/diagnostics.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"

using namespace neoruntime_ipc_sdk;

namespace {

// python's _isolate_env fixture: point every endpoint at a nonexistent
// dir — the probes must report exists=false without raising.
class IsolatedEnv : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string base = testing::TempDir() + "/nope-diagnostics";
        setenv("AI_RUNTIME_ENDPOINT", ("unix://" + base + "/ai.sock").c_str(), 1);
        setenv("EVENT_BUS_ENDPOINT", ("unix://" + base + "/events.sock").c_str(), 1);
        setenv("DEVICE_CONTROL_ENDPOINT", ("unix://" + base + "/device.sock").c_str(), 1);
        setenv("CAMERA_CONTROL_ENDPOINT", ("unix://" + base + "/camera-ctl.sock").c_str(), 1);
        setenv("APP_MANAGER_ENDPOINT", ("unix://" + base + "/appmgr.sock").c_str(), 1);
        setenv("CAMERA_SOCK_PATH", (base + "/camera.sock").c_str(), 1);
        setenv("ENCODED_SOCKET_DIR", (base + "/encoded").c_str(), 1);
    }
};

}  // namespace

TEST(UdsPath, StripsUnixScheme) {
    EXPECT_EQ(uds_path("unix:///run/aipc/x.sock"), "/run/aipc/x.sock");
}

TEST(UdsPath, PlainPathUnchanged) {
    EXPECT_EQ(uds_path("/run/aipc/x.sock"), "/run/aipc/x.sock");
}

TEST_F(IsolatedEnv, StructureAndNeverRaises) {
    DiagnosticsReport report = diagnostics();

    EXPECT_FALSE(report.sdk_version.empty());
    EXPECT_FALSE(report.toolchain.standard.empty());
    const auto cv2 = report.optional_deps.find("cv2");
    ASSERT_NE(cv2, report.optional_deps.end());
    EXPECT_TRUE(cv2->second.available);  // hard link dependency in C++
    EXPECT_FALSE(cv2->second.version.empty());

    EXPECT_FALSE(report.services.ai_runtime.exists);
    EXPECT_FALSE(report.services.ai_runtime.connectable);
    EXPECT_FALSE(report.services.event_bus.exists);
    EXPECT_FALSE(report.services.device_control.exists);
    EXPECT_FALSE(report.services.camera_control.exists);
    EXPECT_FALSE(report.services.app_manager.exists);
    EXPECT_FALSE(report.media.camera_sock.exists);
    EXPECT_TRUE(report.media.streams.empty());
    EXPECT_FALSE(report.accel_router.policy.empty());
}

TEST_F(IsolatedEnv, ListsEncodedSockets) {
    const std::string enc = testing::TempDir() + "/encoded-dx";
    std::filesystem::create_directories(enc);
    for (const char* name : {"main.sock", "sub.sock", "notes.txt"})
        close(open((enc + "/" + name).c_str(), O_CREAT | O_WRONLY, 0644));
    setenv("ENCODED_SOCKET_DIR", enc.c_str(), 1);

    DiagnosticsReport report = diagnostics();
    ASSERT_EQ(report.media.streams.size(), 2u);
    EXPECT_EQ(report.media.streams[0], "main");
    EXPECT_EQ(report.media.streams[1], "sub");
}

TEST_F(IsolatedEnv, CheckHealthyWhenNothingRequired) {
    CheckReport report = check();
    EXPECT_TRUE(report.healthy);
    EXPECT_TRUE(report.problems.empty());
    for (const char* name : {"inference", "camera", "cv2", "dsp"})
        EXPECT_NE(report.checks.find(name), report.checks.end()) << name;
}

TEST_F(IsolatedEnv, CheckFailedRequirementsCarryHints) {
    CheckReport report = check({"inference", "camera_sock"});
    EXPECT_FALSE(report.healthy);
    EXPECT_FALSE(report.checks.at("inference"));
    bool saw_ai_runtime = false, saw_video_frames = false;
    for (const std::string& problem : report.problems) {
        if (problem.find("ai-runtime") != std::string::npos) saw_ai_runtime = true;
        if (problem.find("video frames") != std::string::npos) saw_video_frames = true;
    }
    EXPECT_TRUE(saw_ai_runtime);
    EXPECT_TRUE(saw_video_frames);
    EXPECT_THROW(report.raise_if_unhealthy(), std::runtime_error);
    try {
        report.raise_if_unhealthy();
        FAIL() << "unreachable";
    } catch (const std::runtime_error& exc) {
        EXPECT_NE(std::string(exc.what()).find("health check failed"), std::string::npos);
    }
}

TEST_F(IsolatedEnv, CheckUnknownNameRejected) {
    try {
        check({"quantum_link"});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("unknown check"), std::string::npos);
    }
}

TEST_F(IsolatedEnv, CheckModelAndStreamProbesDownPath) {
    // no daemons on a dev machine: both live probes must land in their
    // catch branches — checks false, lookup errors recorded, problems
    // carry the actionable messages
    InferenceClient inference;
    CameraClient camera;
    CheckReport report = check({"model_registered", "stream_active"}, &inference,
                               "person_v1", &camera, "main");
    EXPECT_FALSE(report.healthy);
    EXPECT_FALSE(report.checks.at("model_registered"));
    EXPECT_FALSE(report.checks.at("stream_active"));
    EXPECT_TRUE(report.model_lookup_error.has_value());
    EXPECT_TRUE(report.stream_lookup_error.has_value());
    bool saw_not_registered = false, saw_not_active = false;
    for (const std::string& problem : report.problems) {
        if (problem.find("not registered") != std::string::npos) saw_not_registered = true;
        if (problem.find("not active") != std::string::npos) saw_not_active = true;
    }
    EXPECT_TRUE(saw_not_registered);
    EXPECT_TRUE(saw_not_active);
}
