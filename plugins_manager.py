import json
import traceback

class PluginManager(object):

    def __init__(self, PROD, CPU_ONLY, logger):
        super(PluginManager, self).__init__()

        self.CPU_ONLY = CPU_ONLY
        self.path = "./resources/app" if PROD else "."
        self.modules_path = "resources.app." if PROD else ""
        self.logger = logger
        self.setupModules = set()
        self.refresh_active_plugins()

    def reset_plugins (self):
        self.plugins = {
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
            "output-audio": {
                "pre": [],
                "post": []
            },
        }

    def refresh_active_plugins (self):

        self.reset_plugins()
        status = []

        with open("plugins.txt") as f:
            lines = f.read().split("\n")

        for line in lines:
            if line.startswith("*"):
                plugin_id = line[1:]
                self.logger.info(f'plugin_id: {plugin_id}')
                try:
                    with open(f'{self.path}/plugins/{plugin_id}/plugin.json') as f:
                        plugin_json = f.read()
                        plugin_json = json.loads(plugin_json)

                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "start", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "start", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "load-model", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "load-model", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "synth-line", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "synth-line", "post"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "output-audio", "pre"], [])
                        self.load_module_function(plugin_json, plugin_id, ["back-end-entry-file", "output-audio", "post"], [])

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
                    def register_function(fn):
                        if fn.__name__==function:
                            self.plugins[structure2[-2]][structure2[-1]].append([plugin_name, file_name, fn])
                        elif fn.__name__=="setup":
                            if f'{plugin_name}/{file_name}' not in self.setupModules:
                                self.setupModules.add(f'{plugin_name}/{file_name}')
                                fn()


                    setup = {"logger": self.logger, "isCPUonly": self.CPU_ONLY}

                    exec(open(f'{self.path}/plugins/{plugin_name}/{file_name}').read(), None, {
                        "setup": setup,
                        "plugins": self.plugins,
                        "register_function": register_function
                    })

                else:
                    self.logger.info(f'[Plugin: {plugin_name}]: Cannot import {file_name} file for {structure2[-1]} {structure2[-2]} entry-point: Only python files are supported right now.')


    def run_plugins (self, plist, event="", data=None):
        if len(plist):
            self.logger.info("Running plugins for event:" + event)
        for [plugin_name, file_name, function] in plist:
            try:
                self.logger.info(plugin_name)
                self.logger.set_logger_prefix(plugin_name)
                function(data)

                self.logger.set_logger_prefix("")
            except:
                self.logger.info(f'[Plugin run error at event "{event}": {plugin_name}]')
                self.logger.info(traceback.format_exc())
