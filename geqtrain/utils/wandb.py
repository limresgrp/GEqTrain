import wandb
import logging
from wandb.util import json_friendly_val

from pathlib import Path

import os, shutil
def make_archive(source, destination):
    '''
    example usage:
    make_archive('/path/to/folder', '/path/to/folder.zip')
    '''
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s'%(name,format), destination)


def upload_zipped_code_on_wandb(source, upload_name):
        '''
        upload_name must be without file ext
        source must be str
        '''
        if isinstance(source, Path):
            source = str(source)

        dest = f"./{upload_name}.zip"
        make_archive(source, dest)
        wandb.save(dest, policy = "now")
        # os.remove(dest) # can't remove since wandb.save call is async

def init_n_update(config):
    conf_dict = dict(config)
    # wandb mangles keys (in terms of type) as well, but we can't easily correct that because there are many ambiguous edge cases. (E.g. string "-1" vs int -1 as keys, are they different config keys?)
    if any(not isinstance(k, str) for k in conf_dict.keys()):
        raise TypeError(
            "Due to wandb limitations, only string keys are supported in configurations."
        )

    # download from wandb set up
    config.run_id = wandb.util.generate_id()

    wandb.init(
        project=config.wandb_project,
        config=conf_dict,
        name=config.run_name,
        notes=conf_dict.get('experiment_description', None),
        resume="allow",
        id=config.run_id,
    )

    # save config as-is on wandb for experiment relaunching

    wandb.save(Path(config.filepath).resolve(), policy = "now")

    # upload geqtrain code

    source = Path(__file__).parent.parent.resolve()
    upload_zipped_code_on_wandb(source, 'geqtrain_source_code')

    # upload ad-hoc code

    if 'code_folder_name' in config:
        source = Path().resolve() / config['code_folder_name']
        upload_zipped_code_on_wandb(source, f'{config["code_folder_name"]}_source_code')

    # download from wandb set up

    updated_parameters = dict(wandb.config)
    for k, v_new in updated_parameters.items():
        skip = False
        if k in config.keys():
            # double check the one sanitized by wandb
            v_old = json_friendly_val(config[k])
            if repr(v_new) == repr(v_old):
                skip = True
        if skip:
            logging.info(f"# skipping wandb update {k} from {v_old} to {v_new}")
        else:
            config.update({k: v_new})
            logging.info(f"# wandb update {k} from {v_old} to {v_new}")
    return config


def resume(config):
    # resume to the old wandb run
    wandb.init(
        project=config.wandb_project,
        resume="must",
        id=config.run_id,
    )
