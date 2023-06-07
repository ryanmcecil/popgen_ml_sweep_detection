from typing import Dict

from simulate.popgen_simulators import retrieve_simulator


def retrieve_image_width_from_settings(settings: Dict):
    """Returns final image width dependent on conversion settings

    Parameters
    ----------
    settings (Dict) - Conversion and simulation configuration settings.

    Returns
    -------
    int: Image width after conversions

    """
    if settings['conversions'][0]['conversion_type'] == 'imagene':
        return 128
    elif settings['conversions'][0]['conversion_type'] == 'zero_padding_imagene':
        return retrieve_max_width_from_settings(settings)
    else:
        raise NotImplementedError


def retrieve_max_width_from_settings(settings: Dict, use_min: bool = False):
    """Retrieves maximum width across all simulated types

    Parameters
    ----------
    settings (Dict) - Conversion and simulation configuration settings.

    Returns
    -------
    int: Maximum width across all simulation types

    """
    # Check for multiple population data
    if 'pop' in settings['conversions'][0]:
        pop = settings['conversions'][0]['pop']
    else:
        pop = None

    # Compute widths across different simulation types and take maximum
    widths = []
    for label in 'sweep', 'neutral':
        config = settings['simulations'][label][0]
        if use_min:
            widths.append(retrieve_simulator(config['software'])(
                config).retrieve_max_or_min_width(pop=pop, get_max=False))
        else:
            widths.append(retrieve_simulator(config['software'])(
                config).retrieve_max_or_min_width(pop=pop, get_max=True))

    if use_min:
        return min(widths)
    else:
        return max(widths)
