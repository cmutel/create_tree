# -*- coding: utf-8 -*-
import click
import os
from brightway2 import *


@click.command()
@click.option('--version', help='Ecoinvent version (2 or 3)', type=int)
@click.option('--path', help='Directory path for ecoinvent input data')
def main(version, path):
    """Setup base projects for ecoinvent 2.2 and 3.3."""
    if version not in (2,3):
        print("Need valid version number")
        return

    if not path or not os.path.isdir(path):
        print("This is not a valid path: {}".format(path))
        return

    project_name = "2.2 tree base" if version == 2 else "3.3 tree base"

    if project_name in projects:
        print("This project already exists: {}".format(project_name))
        return

    projects.set_current(project_name)
    bw2setup()

    if version == 2:
        ei22 = SingleOutputEcospold1Importer(
            path,
            'ecoinvent')
        ei22.apply_strategies()
        ei22.write_database()
    else:
        ei33cu = SingleOutputEcospold2Importer(
            path,
            'ecoinvent')
        ei33cu.apply_strategies()
        ei33cu.write_database()

    print("Created project: {}".format(project_name))


if __name__ == "__main__":
    main()
