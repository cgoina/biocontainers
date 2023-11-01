import dask.array as da
import functools
import nrrd
import numpy as np
import os
import tifffile

from . import zarr_utils


def save(data, container_path, subpath,
         blocksize=None,
         resolution=None,
         scale_factors=None,
         dask_cluster=None,
):
    """
    Persist distributed data - typically a dask array to the specified
    container

    Parameters
    ==========
    data - the dask array that needs 
    """
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]
    persist_block = None
    if container_ext == '.nrrd':
        print(f'Persist data as nrrd {container_path} ({real_container_path})',
              flush=True)
        output_dir = os.path.dirname(container_path)
        output_name = os.path.basename(path_comps[0])
        persist_block = functools.partial(_save_block_to_nrrd,
                                          output_dir=output_dir,
                                          output_name=output_name,
                                          ext=container_ext)
    elif container_ext == '.tif' or container_ext == '.tiff':
        print(f'Persist data as tiff {container_path} ({real_container_path})',
              flush=True)
        output_dir = os.path.dirname(container_path)
        output_name = os.path.basename(path_comps[0])
        persist_block = functools.partial(_save_block_to_tiff,
                                          output_dir=output_dir,
                                          output_name=output_name,
                                          resolution=resolution,
                                          ext=container_ext)
    elif container_ext == '.n5' or (container_ext == '' and subpath):
        print(f'Persist data as n5 {container_path} ',
              f'({real_container_path}):{subpath}',
              flush=True)
        output_data = zarr_utils.create_dataset(
            container_path,
            subpath,
            data.shape,
            blocksize,
            data.dtype,
            data_store_name='n5',
            pixelResolution=resolution,
            downsamplingFactors=scale_factors,
        )
        persist_block = functools.partial(_save_block_to_zarr, output=output_data)
    elif container_ext == '.zarr':
        print(f'Persist data as zarr {container_path} ',
              f'({real_container_path}):{subpath}',
              flush=True)
        output_data = zarr_utils.create_dataset(
            container_path,
            subpath,
            data.shape,
            blocksize,
            data.dtype,
            data_store_name='zarr',
            pixelResolution=resolution,
            downsamplingFactors=scale_factors,
        )
        persist_block = functools.partial(_save_block_to_zarr, output=output_data)

    else:
        print(f'Cannot persist data using {container_path} ',
              f'({real_container_path}): {subpath}',
              flush=True)

    if persist_block is not None:
        save_blocks(data, persist_block, dask_cluster=dask_cluster)
        print(f'Finished writing data to {container_path}:{subpath}',
              flush=True)


def save_blocks(dimage, persist_block, dask_cluster=None):
    persisted_array = da.map_blocks(persist_block, dimage,
                                    dtype=dimage.dtype,
                                    meta=np.array(dimage.shape))
    res = dask_cluster.compute(persisted_array)
    dask_cluster.gather([res])


def _save_block_to_nrrd(block, output_dir=None, output_name=None,
                        block_info=None,
                        ext='.nrrd'):
    if block_info is not None:
        block_coords = tuple([slice(c[0],c[1])
                              for c in block_info[0]['array-location']])

        saved_blocks_count = np.prod(block_info[None]['num-chunks'])
        if saved_blocks_count > 1:
            filename = (output_name + '-' +
                        '-'.join(map(str, block_info[0]['chunk-location'])) +
                        ext)
        else:
            filename = output_name + ext

        full_filename = os.path.join(output_dir, filename)
        print(f'Write block {block.shape}',
              f'block_info: {block_info}',
              f'block_coords: {block_coords}',
              flush=True)
        nrrd.write(full_filename, block.transpose(2, 1, 0),
                   compression_level=2)
    return block


def _save_block_to_tiff(block, output_dir=None, output_name=None,
                        block_info=None,
                        resolution=None,
                        ext='.tif',
                        ):
    if block_info is not None:
        block_coords = tuple([slice(c[0],c[1])
                              for c in block_info[0]['array-location']])

        saved_blocks_count = np.prod(block_info[None]['num-chunks'])
        if saved_blocks_count > 1:
            filename = (output_name + '-' +
                        '-'.join(map(str, block_info[0]['chunk-location'])) +
                        ext)
        else:
            filename = output_name + ext

        full_filename = os.path.join(output_dir, filename)
        print(f'Write block {block.shape}',
              f'block_info: {block_info}',
              f'block_coords: {block_coords}',
              f'to {full_filename}',
              flush=True)
        tiff_metadata = {
            'axes': 'ZYX',
        }
        if resolution is not None:
            tiff_metadata['resolution'] = resolution
        tifffile.imwrite(full_filename, block, 
                         metadata=tiff_metadata)
    return block


def _save_block_to_zarr(block, output=None, block_info=None):
    if block_info is not None:
        block_coords = tuple([slice(c[0],c[1]) 
                              for c in block_info[0]['array-location']])
        print(f'Write block {block.shape}',
              f'block_info: {block_info}',
              f'block_coords: {block_coords}',
              flush=True)
        output[block_coords] = block
    return block
