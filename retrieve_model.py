from models.popgen_mlmodels import retrieve_ml_model
from models.popgen_summary_statistics import SummaryStatPopGenModel


def retrieve_model(config):
    if config['model']['type'] == 'ml':
        return retrieve_ml_model(config['model']['name'])
    elif config['model']['type'] == 'statistic':
        return SummaryStatPopGenModel
    else:
        raise NotImplementedError