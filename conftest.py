"""Pytest configuration to disable conflicting ROS plugins."""


def pytest_configure(config):
    # Unregister ROS plugins that conflict with standard pytest
    for plugin_name in [
        "launch_testing_ros_pytest_entrypoint",
        "launch_testing",
    ]:
        plugin = config.pluginmanager.get_plugin(plugin_name)
        if plugin is not None:
            config.pluginmanager.unregister(plugin)
