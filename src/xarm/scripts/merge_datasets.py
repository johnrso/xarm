import click
import os
import multiprocessing as mp
# make a click function that takes in a list of folders. it then merges the files inside without overwriting. to do this, rename each file to traj_x, where x increments by 1 each time.
@click.command()
@click.argument('folders', nargs=-1)
@click.option('--output', '-o', default='merged', help='Name of output folder')
def merge(folders, output):
    if os.path.exists(output):
        raise ValueError(f'{output} already exists. Please delete it first.')
    os.makedirs(output, exist_ok=True)

    i = 0

    args = []
    for folder in folders:
        # get all subfolders
        for file in os.listdir(folder):
            if "conv" not in file:
                old_fn = os.path.join(folder, file)
                new_fn = os.path.join(output, file)
                args.append((old_fn, new_fn))
                i += 1

    print(f'About to merge {i} files into {output}.')
    with mp.Pool(10) as pool:
        pool.starmap(copy_file, args)
    print(f'Finished merging {i} files into {output}')
    exit()

def copy_file(old_fn, new_fn):
    # create a symlink to the file
    try:
        os.symlink(old_fn, new_fn)
    except FileExistsError:
        pass

if __name__ == '__main__':
    merge()
