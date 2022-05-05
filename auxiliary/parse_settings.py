from configobj import ConfigObj, flatten_errors
from validate import Validator, VdtValueError, VdtTypeError
from os.path import exists
from os.path import abspath
from os.path import dirname
import warnings

def parse_settings(algorithm, racetrack, visualize):
    """parse the algorithm settings defined by the user"""

    # load settings file
    dirpath = dirname(dirname(abspath(__file__)))
    path = dirpath + '/racetracks/' + racetrack + '/settings/' + algorithm + '.cfg'
    path_config = dirpath + '/settings/' + algorithm + '.cfg'

    if not exists(path_config):
        msg = 'File /settings/%s.cfg defining the valid algorithm settings is missing for this algorithm!' % algorithm
        raise Exception(msg)

    if not exists(path):
        msg = 'No *.cfg file with settings found for algorithm %s and racetrack %s.\n' % (algorithm, racetrack)
        msg += 'Using default settings.'
        warnings.warn(msg)

    # parse settings
    vdt = Validator()
    config = ConfigObj(path, configspec=path_config)
    results = config.validate(vdt, preserve_errors=True)

    # display potential error messages for wrong settings
    for entry in flatten_errors(config, results):

        [sectionList, key, error] = entry
        if error == False:
            msg = "The parameter %s was not in the config file\n" % key
            msg += "Please check to make sure this parameter is present and there are no mis-spellings."
            raise Exception(msg)

        if key is not None:
            if isinstance(error, VdtValueError) or isinstance(error, VdtTypeError):
                optionString = config.configspec[key]
                msg = "The parameter %s was set to %s which is not one of the allowed values\n" % (key, config[key])
                msg += "Please set the value to be in %s" % optionString
                raise Exception(msg)

    # store settings in a dictionary
    settings = dict(config)

    # check if the user specified unnecessary settings
    keys = settings.keys()

    for k in keys:
        if k not in config.default_values.keys():
            msg = 'Setting "%s" is not defined for algorithm %s' % (k, algorithm)
            warnings.warn(msg)

    # add path to the optimal raceline and visualization flag
    settings['path_raceline'] = 'racetracks/' + racetrack + '/raceline.csv'
    settings['VISUALIZE'] = visualize
    settings['RACETRACK'] = racetrack

    return settings