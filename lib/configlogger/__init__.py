'''
Utility for logging experimental configuration.
'''

import codecs
from datetime import datetime
from collections import OrderedDict

def writeConfig(file_path, settings, title=None):
    '''Write an experimental configuration to a file.

    Always writes the current date and time at the head of the file.

    The optional title argument is a string to write at the head of the file,
        before date and time.

    Settings should be passed in as a list, in the desired order for writing.
    To write the value of a single setting, pass it as (name, value) pair.
    To group several settings under a section, pass a (name, dict) pair, where
        the first element is the name of the section, and the second is a
        dict (or OrderedDict) of { setting: value } format.

    For example, the following call:
        configlogger.writeConfig(some_path, [
            ('Value 1', 3),
            ('Some other setting', True),
            ('Section 1', OrderedDict([
                ('sub-value A', 12.4),
                ('sub-value B', 'string')
            ]))
        ], title='My experimental configuration')
    will produce the following configuration log:
        
        My experimental configuration
        Run time: 1970-01-01 00:00:00

        Value 1: 3
        Some other setting: True

        ## Section 1 ##
        sub-value A: 12.4
        sub-value B: string
    '''

    group_set = set([dict, OrderedDict])

    with codecs.open(file_path, 'w', 'utf-8') as stream:
        # headers
        if title:
            stream.write('%s\n' % title)
        stream.write('Run time: %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        stream.write('\n')

        for (key, value) in settings:
            if type(value) in group_set:
                stream.write('\n## %s ##\n' % key)
                for (sub_key, sub_value) in value.items():
                    stream.write('%s: %s\n' % (sub_key, str(sub_value)))
                stream.write('\n')
            else:
                stream.write('%s: %s\n' % (key, str(value)))
