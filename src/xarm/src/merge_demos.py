import click
import os
import shutil
import multiprocessing as mp

@click.command()
@click.argument('folders', nargs=-1)
@click.option('--output', '-o', default='merged', help='Name of output folder')
@click.option('--num_per_folder', '-n', default=-1, help='Number of files to take from each folder')
def merge(folders, output, num_per_folder):
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    token_folder = os.path.join(output, 'none')
    if not os.path.exists(token_folder):
        os.makedirs(token_folder)
    i = 0

    args = []
    for folder in folders:
        for j, file in enumerate(os.listdir(folder)):
            if j == num_per_folder:
                break
            if file.endswith(".h5"):
                old_fn = os.path.join(folder, file)
                new_fn = os.path.join(token_folder, f'traj_{i}.h5')
                args.append((old_fn, new_fn))
                i += 1

    print(f'About to merge {i} files into {output}.')
    with mp.Pool(10) as pool:
        pool.starmap(copy_file, args)
    print(f'Finished merging {i} files into {output}')

def copy_file(old_fn, new_fn):
    shutil.copyfile(old_fn, new_fn)

if __name__ == '__main__':
    merge()
