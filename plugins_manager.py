import json
import traceback


class PluginManager(object):

    def __init__(self, APP_VERSION, PROD, CPU_ONLY, logger):
        super(PluginManager, self).__init__()

        self.APP_VERSION = APP_VERSION
        self.CPU_ONLY = CPU_ONLY
        self.PROD = PROD
        self.path = "./resources/app" if PROD else "."
        self.modules_path = "resources.app." if PROD else ""
        self.logger = logger
        self.setupModules = set()
        self.enabledPlugins = set()
        self.teardownModules = {}
        self.refresh_active_plugins()

    def reset_plugins (self):
        self.plugins = {
            "custom-event": [],
            "start": {
                "pre": [],
                "post": []
            },
            "load-model": {
                "pre": [],
                "post": []
            },
            "synth-line": {
                "pre": [],
                "post": []
            },
            "batch-synth-line": {
                "pre": [],
                "post": []
            },
            "output-audio": {
                "pre": [],
                "post": []
            },
        }

    def refresh_active_plugins (self):

        with open("plugins.txt") as f:
            lines = f.read().split("\n")

        removed_plugins = []
        for line in lines:
            if not line.startswith("*") and line in self.enabledPlugins:
                removed_plugins.append(line)

        for plugin_id in removed_plugins:
            if plugin_id in self.teardownModules:
                for func in self.teardownModules[plugin_id]:
                    params = {"logger": self.logger, "appVersion": self.APP_VERSION, "isCPUonly": self.CPU_ONLY, "isDev": not self.PROD}
                    func(params)

        self.reset_plugins()
        status = []


        for line in lines:
            if line.startswith("*"):
                plugin_id = line[1:]
                self.logger.info(f'plugin_id: {plugin_id}')
                try:
                    with open(f'{self.path}/plugins/{plugin_id}/plugin.json') as f:
                        plugin_json = f.read()
                        plugin_json = json.loads(plugin_json)

                        minVersionOk = checkVersionRequirements(plugin_json["min-app-version"] if "min-app-version" in plugin_json else None, self.APP_VERSION)
                        maxVersionOk = checkVersionRequirements(plugin_json["max-app-version"] if "max-app-version" in plugin_json else None, self.APP_VERSION, True)

                        if not minVersionOk or not maxVersionOk:
                            continue

                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "start", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "start", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "load-model", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "load-model", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "synth-line", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "synth-line", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "batch-synth-line", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "batch-synth-line", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "output-audio", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "output-audio", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-hooks", "custom-event"], [])

                        self.enabledPlugins.add(plugin_id)

                    status.append("OK")
                except:
                    self.logger.info(traceback.format_exc())
                    status.append(plugin_id)

        return status

    def load_module_function (self, plugin_json, plugin_name, structure, structure2):

        if structure[0] in plugin_json and plugin_json[structure[0]] is not None:
            key = structure[0]
            structure2.append(key)
            plugin_json = plugin_json[key]
            del structure[0]
            if len(structure):
                return self.load_module_function(plugin_json, plugin_name, structure, structure2)
            else:
                file_name = plugin_json["file"]
                function = plugin_json["function"]

                if not file_name:
                    return

                if file_name.endswith(".py"):
                    setup = {"logger": self.logger, "appVersion": self.APP_VERSION, "isCPUonly": self.CPU_ONLY, "isDev": not self.PROD}

                    with open(f'{self.path}/plugins/{plugin_name}/{file_name}') as f:

                        locals_data = locals()
                        locals_data["setupData"] = setup
                        locals_data["plugins"] = self.plugins

                        exec(f.read(), None, locals_data)

                        import types
                        for key in list(locals_data.keys()):
                            if isinstance(locals_data[key], types.FunctionType):
                                if key==function:
                                    if structure2[-1]=="custom-event":
                                        self.plugins[structure2[-1]].append([plugin_name, file_name, locals_data[key]])
                                    else:
                                        self.plugins[structure2[-2]][structure2[-1]].append([plugin_name, file_name, locals_data[key]])
                                elif key=="setup":
                                    if f'{plugin_name}/{file_name}' not in self.setupModules:
                                        self.setupModules.add(f'{plugin_name}/{file_name}')
                                        locals_data[key](setup)
                                elif key=="teardown":
                                    if plugin_name not in self.teardownModules.keys():
                                        self.teardownModules[plugin_name] = []
                                    self.teardownModules[plugin_name].append(locals_data[key])


                else:
                    self.logger.info(f'[Plugin: {plugin_name}]: Cannot import {file_name} file for {structure2[-1]} {structure2[-2]} entry-point: Only python files are supported right now.')


    def run_plugins (self, plist, event="", data=None):
        response = None
        if event=="custom-event" and "pluginId" not in data:
            self.logger.info(f'Custom event called, but "pluginId" not specified. Not running.')
            return None

        if len(plist):
            self.logger.info("Running plugins for event:" + event)

        for [plugin_name, file_name, function] in plist:

            if event=="custom-event" and plugin_name!=data["pluginId"]:
                continue

            try:
                self.logger.info(plugin_name)
                self.logger.set_logger_prefix(plugin_name)
                function(data)

                self.logger.set_logger_prefix("")
            except:
                self.logger.info(f'[Plugin run error at event "{event}": {plugin_name}]')
                self.logger.info(traceback.format_exc())

        return response


def checkVersionRequirements (requirements, appVersion, checkMax=False):

    if not requirements:
        return True

    appVersionRequirement = [int(val) for val in str(requirements).split(".")]
    appVersionInts = [int(val) for val in str(appVersion).split(".")]
    appVersionOk = True

    if checkMax:

        if appVersionRequirement[0] >= appVersionInts[0]:
            if len(appVersionRequirement)>1 and int(appVersionRequirement[0])==appVersionInts[0]:
                if appVersionRequirement[1] >= appVersionInts[1]:
                    if len(appVersionRequirement)>2 and int(appVersionRequirement[1])==appVersionInts[1]:
                        if appVersionRequirement[2] >= appVersionInts[2]:
                            pass
                        else:
                            appVersion = False
                else:
                    appVersionOk = False
        else:
            appVersionOk = False

    else:

        if appVersionRequirement[0] <= appVersionInts[0]:
            if len(appVersionRequirement)>1 and int(appVersionRequirement[0])==appVersionInts[0]:
                if appVersionRequirement[1] <= appVersionInts[1]:
                    if len(appVersionRequirement)>2 and int(appVersionRequirement[1])==appVersionInts[1]:
                        if appVersionRequirement[2] <= appVersionInts[2]:
                            pass
                        else:
                            appVersion = False
                else:
                    appVersionOk = False
        else:
            appVersionOk = False

    return appVersionOk
