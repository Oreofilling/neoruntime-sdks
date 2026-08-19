// test_config.cpp — GoogleTest mirror of python/tests/test_config.py.
//
// Verifies the Config accessors read the same env vars and fall back to the
// same defaults as the Python SDK, so a deployment can configure both runtimes
// identically. Each test scopes its env mutation with an RAII guard so the
// process environment is restored regardless of assertion outcome.
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "hailo_ipc_sdk/config.hpp"

namespace {

// RAII save/restore for a single environment variable. Unsetting in the ctor
// gives the test a clean baseline (matches the Python `del os.environ[...]`
// idiom); the dtor restores the original value (or re-unsets if it was unset).
class EnvScope {
public:
    explicit EnvScope(const char* var) : var_(var) {
        const char* cur = std::getenv(var);
        was_set_ = (cur != nullptr);
        if (was_set_) saved_ = cur;
        ::unsetenv(var);
    }
    ~EnvScope() {
        if (was_set_) ::setenv(var_.c_str(), saved_.c_str(), /*overwrite=*/1);
        else           ::unsetenv(var_.c_str());
    }
    void set(const std::string& value) const { ::setenv(var_.c_str(), value.c_str(), 1); }
    EnvScope(const EnvScope&) = delete;
    EnvScope& operator=(const EnvScope&) = delete;
private:
    std::string var_;
    bool was_set_ = false;
    std::string saved_;
};

}  // namespace

// ---- app id ----------------------------------------------------------------
TEST(Config, AppIdDefault) {
    EnvScope scope("APP_ID");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_app_id(), "unknown");
}

TEST(Config, AppIdFromEnv) {
    EnvScope scope("APP_ID");
    scope.set("my-test-app");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_app_id(), "my-test-app");
}

// Empty-string env values are treated as unset (matches Config::env contract).
TEST(Config, AppIdEmptyEnvFallsBack) {
    EnvScope scope("APP_ID");
    scope.set("");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_app_id(), "unknown");
}

// ---- endpoints -------------------------------------------------------------
TEST(Config, InferenceEndpointDefault) {
    EnvScope scope("AI_RUNTIME_ENDPOINT");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_inference_endpoint(),
              "unix:///run/aipc/ai-runtime.sock");
}

TEST(Config, InferenceEndpointFromEnv) {
    EnvScope scope("AI_RUNTIME_ENDPOINT");
    scope.set("unix:///custom/ai-runtime.sock");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_inference_endpoint(),
              "unix:///custom/ai-runtime.sock");
}

TEST(Config, EventBusEndpointDefault) {
    EnvScope scope("EVENT_BUS_ENDPOINT");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_event_bus_endpoint(),
              "unix:///run/aipc/event-bus.sock");
}

TEST(Config, DeviceControlEndpointDefault) {
    EnvScope scope("DEVICE_CONTROL_ENDPOINT");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_device_control_endpoint(),
              "unix:///run/aipc/device-control.sock");
}

TEST(Config, CameraControlEndpointDefault) {
    EnvScope scope("CAMERA_CONTROL_ENDPOINT");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_camera_control_endpoint(),
              "unix:///run/aipc/camera-control.sock");
}

TEST(Config, AppManagerEndpointDefault) {
    EnvScope scope("APP_MANAGER_ENDPOINT");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_app_manager_endpoint(),
              "unix:///run/aipc/app-manager.sock");
}

// ---- filesystem paths ------------------------------------------------------
TEST(Config, ShmBasePathDefault) {
    EnvScope scope("SHM_BASE_PATH");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_shm_base_path(), "/run/aipc/shm");
}

TEST(Config, EncodedSocketDirDefault) {
    EnvScope scope("ENCODED_SOCKET_DIR");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_encoded_socket_dir(), "/run/aipc/encoded");
}

// ---- path translation ------------------------------------------------------
TEST(Config, TranslateOptAipcToHost) {
    EnvScope scope("AIPC_HOST_PREFIX");
    scope.set("/data/aipc");
    EXPECT_EQ(hailo_ipc_sdk::Config::translate_path_to_host("/opt/aipc/models/foo"),
              "/data/aipc/models/foo");
}

TEST(Config, TranslateDataAipcToHost) {
    EnvScope scope("AIPC_HOST_PREFIX");
    scope.set("/data/aipc");
    EXPECT_EQ(hailo_ipc_sdk::Config::translate_path_to_host("/data/aipc/run/bar.sock"),
              "/data/aipc/run/bar.sock");
}

TEST(Config, TranslateUnknownPrefixUnchanged) {
    EnvScope scope("AIPC_HOST_PREFIX");
    scope.set("/data/aipc");
    EXPECT_EQ(hailo_ipc_sdk::Config::translate_path_to_host("/tmp/elsewhere"),
              "/tmp/elsewhere");
}

// ---- debug / log level -----------------------------------------------------
TEST(Config, IsDebugDefaultFalse) {
    EnvScope scope("DEBUG");
    EXPECT_FALSE(hailo_ipc_sdk::Config::is_debug());
}

TEST(Config, IsDebugTrueWhenOne) {
    EnvScope scope("DEBUG");
    scope.set("1");
    EXPECT_TRUE(hailo_ipc_sdk::Config::is_debug());
}

TEST(Config, IsDebugFalseWhenNotOne) {
    EnvScope scope("DEBUG");
    scope.set("true");  // only literal "1" is truthy, mirroring the Python check
    EXPECT_FALSE(hailo_ipc_sdk::Config::is_debug());
}

TEST(Config, LogLevelDefault) {
    EnvScope scope("LOG_LEVEL");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_log_level(), "INFO");
}

TEST(Config, LogLevelFromEnv) {
    EnvScope scope("LOG_LEVEL");
    scope.set("DEBUG");
    EXPECT_EQ(hailo_ipc_sdk::Config::get_log_level(), "DEBUG");
}
