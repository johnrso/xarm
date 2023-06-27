import click
import os
import multiprocessing as mp
import datetime

@click.command()
@click.argument('folders', nargs=-1)
@click.option('--output', '-o', default='merged', help='Name of output folder')
@click.option('--num_per_folder', '-n', default=-1, help='Number of files to take from each folder')
def merge(folders, output, num_per_folder):
    if os.path.exists(output):
        raise ValueError(f'{output} already exists. Please delete it first.')
    os.makedirs(output, exist_ok=True)


    log_file = os.path.join(output, 'log.txt')
    msg = f'About to merge {len(folders)} folders into {output}.'
    msg += f'\nFolders:'
    for folder in folders:
        msg += f'\n{folder}'
    msg = f'{datetime.datetime.now()}: {msg}'
    print(msg)
    with open(log_file, 'w') as f:
        f.write(msg)

    i = 0

    args = []
    for folder in folders:
        for j, file in enumerate(os.listdir(folder)):
            if j == num_per_folder:
                break
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
