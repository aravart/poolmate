class Options(dict):
    def __getattr__(self, name):
        return self[name]


class OptionsBuilder(object):
    def __init__(self, options={}, derived_options=[]):
        self.options = options
        self.derived_options = derived_options

    def __getattr__(self, name):
        options = dict(self.options)
        derived_options = self.derived_options

        def set_attribute(val):
            options[name] = val
            return OptionsBuilder(options, derived_options)
        return set_attribute

    def derived(self, name, func):
        derived_options = list(self.derived_options)
        derived_options.append((name, func))
        return OptionsBuilder(self.options, derived_options)

    def build(self):
        options = Options(self.options)
        for name, func in self.derived_options:
            options[name] = func(options)
        return options
