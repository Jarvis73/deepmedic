import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def read_nii(file_name, out_dtype=np.int16, special=False, only_header=False):
    nib_vol = nib.load(str(file_name))
    vh = nib_vol.header
    if only_header:
        return vh
    affine = vh.get_best_affine()
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    data = nib_vol.get_fdata().astype(out_dtype).transpose(*trans[::-1])
    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)
    return vh, data


def write_nii(data, header, out_path, out_dtype=np.int16, special=False, affine=None):
    if header is not None:
        affine = header.get_best_affine()
    assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    trans_bk = [np.argwhere(np.array(trans[::-1]) == i)[0][0] for i in range(3)]

    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:  # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:  # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:  # Increase z from Interior to Superior
        data = np.flip(data, axis=0)

    out_image = np.transpose(data, trans_bk).astype(out_dtype)
    if header is None and affine is not None:
        out = nib.Nifti1Image(out_image, affine=affine)
    else:
        out = nib.Nifti1Image(out_image, affine=None, header=header)
    nib.save(out, str(out_path))


def zscore(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data.astype(np.float32) - mean) / std


def process_case(nf_case, out_file):
    vh, data = read_nii(nf_case)
    new_data = zscore(data)
    vh['datatype'] = 16     # 16 denotes DT_FLOAT
    write_nii(new_data, vh, out_file, out_dtype=new_data.dtype)
    print(out_file)


def process_all_cases(nf_dir, out_dir, num_workers=8):
    nf_cases = list(nf_dir.glob("volume-*"))
    out_files = [out_dir / f"subMeanDivStd-{nf_case.name}" for nf_case in nf_cases]

    with Pool(num_workers) as p:
        p.starmap(process_case, zip(nf_cases, out_files))


# /////////////////////////////////////////////////////////////////////////////

def merging_case(nf_case, out_file):
    vh, data = read_nii(nf_case, np.uint8)
    new_data = np.clip(data, 0, 1)
    write_nii(new_data, vh, out_file, out_dtype=new_data.dtype)
    print(out_file)
    # if np.sum(new_data) == 0:
    #     print(out_file)

def merge_label_objects(nf_dir, out_dir, num_workers=8):
    nf_cases = list(nf_dir.glob("segmentation-*"))
    out_files = [out_dir / f"merged-label-{nf_case.name}" for nf_case in nf_cases]

    with Pool(num_workers) as p:
        p.starmap(merging_case, zip(nf_cases, out_files))


if __name__ == "__main__":
    nf_dir = Path(__file__).parents[2] / "backup/j1/MIS/data/NF/nii_NF"
    out_dir = Path(__file__).parents[2] / "backup/j1/MIS/data/NF/subMeanDivStd_NF"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Subtract mean and divide standard.
    # process_all_cases(nf_dir, out_dir, num_workers=8)

    # Merge label objects into one foreground class
    out_dir = Path(__file__).parents[2] / "backup/j1/MIS/data/NF/merged_label_NF"
    out_dir.mkdir(parents=True, exist_ok=True)
    merge_label_objects(nf_dir, out_dir, num_workers=8)

