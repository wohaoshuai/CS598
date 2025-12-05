import os
import glob
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from random import choice
from torch.utils.data import DataLoader
from .datasets import MIMICCXR, EHRdataset, MIMIC_CXR_EHR


R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

# chunk-wise reading
def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


# CXR dataset utils
def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    train_transforms.append(transforms.Resize(args.resize))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(transforms.ToTensor())
     

    test_transforms = []
    test_transforms.append(transforms.Resize(args.resize))
    test_transforms.append(transforms.CenterCrop(224))
    test_transforms.append(transforms.ToTensor())

    return train_transforms, test_transforms



class RandomCrop(object):
    "Randomly crop an image"
    
    def __call__(self, sample):
        resize = 256
        random_crop_size = int(np.random.uniform(0.6*resize,resize,1))
        sample=transforms.RandomCrop(random_crop_size)(sample)
        return sample
    
    

class RandomColorDistortion(object):
    "Apply random color distortions to the image"
    
    def __call__(self, sample):
        resize=256

        # Random color distortion
        strength = 1.0 # 1.0 imagenet setting and CIFAR uses 0.5
        brightness = 0.8 * strength 
        contrast = 0.8 * strength
        saturation = 0.8 * strength
        hue = 0.2 * strength
        prob = np.random.uniform(0,1,1) 
        if prob < 0.8:
            sample=transforms.ColorJitter(brightness, contrast, saturation, hue)(sample)

        # Random Grayscale
        sample=transforms.RandomGrayscale(p=0.2)(sample)
        return sample 
    


def get_transforms_simclr(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    # Resize all images to same size, then randomly crop and resize again
    train_transforms.append(transforms.Resize([args.resize, args.resize]))
    # Random affine
    train_transforms.append(transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)))
    # Random crop
    train_transforms.append(RandomCrop())
    # Resize again
    # train_transforms.append(transforms.Resize([args.resize, args.resize], interpolation=3))
    train_transforms.append(transforms.Resize([224, 224], interpolation=3))
    # Random horizontal flip 
    train_transforms.append(transforms.RandomHorizontalFlip())
    # Random color distortions
    train_transforms.append(RandomColorDistortion())
    # Convert to tensor
    train_transforms.append(transforms.ToTensor())
    
    test_transforms = []
    # Resize all images to same size, then center crop and resize again
    test_transforms.append(transforms.Resize([args.resize, args.resize]))
    crop_proportion=0.875
    test_transforms.append(transforms.CenterCrop([int(0.875*args.resize), int(0.875*args.resize)]))
    # test_transforms.append(transforms.Resize([args.resize, args.resize], interpolation=3))
    test_transforms.append(transforms.Resize([224, 224], interpolation=3))
    #Convert to tensor
    test_transforms.append(transforms.ToTensor())

    return train_transforms, test_transforms




def get_cxr_datasets(args):
    """
    Modified to use image paths from medmod_pairs
    """
    # Load medmod_pairs to get actual image paths
    medmod_pairs = pd.read_csv("./medmod_pairs.csv")
    medmod_pairs['dicom_id'] = medmod_pairs['image_path'].apply(
        lambda x: x.split('/')[-1].replace('.jpg', '')
    )
    
    # Create image_path_map: dicom_id -> image_path
    image_path_map = dict(zip(medmod_pairs['dicom_id'], medmod_pairs['image_path']))
    
    # Get all unique image paths
    all_paths = medmod_pairs['image_path'].unique().tolist()
    
    # Determine split from episode_file
    def get_split_from_episode(episode_file):
        if '/train/' in episode_file:
            return 'train'
        elif '/val/' in episode_file:
            return 'val'
        elif '/test/' in episode_file:
            return 'test'
        return None
    
    medmod_pairs['split'] = medmod_pairs['episode_file'].apply(get_split_from_episode)
    
    # Get transforms
    train_transform, test_transform = get_transforms(args)
    
    # Create datasets with image_path_map
    cxr_train_ds = MIMICCXR(all_paths, args, transform=train_transform, split='train', image_path_map=image_path_map)
    cxr_val_ds = MIMICCXR(all_paths, args, transform=test_transform, split='val', image_path_map=image_path_map)
    cxr_test_ds = MIMICCXR(all_paths, args, transform=test_transform, split='test', image_path_map=image_path_map)
    
    return cxr_train_ds, cxr_val_ds, cxr_test_ds




############################################################################

# EHR utils
def get_datasets(discretizer, normalizer, args):
  
    transform = None
    train_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/train'), transforms=transform)
    val_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/val_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/train'), transforms = transform)
    test_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/test_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/test'), transforms = transform)
    return train_ds, val_ds, test_ds


def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
    train_ds, val_ds, test_ds = get_datasets(discretizer, normalizer, dataset_dir)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False, collate_fn=my_collate_ehr, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate_ehr, pin_memory=True, num_workers=16)
    return train_dl, val_dl
        
def my_collate_ehr(batch):
    x = [item[0] for item in batch]
    x, seq_length = pad_zeros(x)
    targets = np.array([item[1] for item in batch])
    return [x, targets, seq_length]

def pad_zeros(arr, min_length=None):

    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length



############################################################################

# Fusion utils
def get_all_metadata(args):
    cxr_metadata = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{args.ehr_data_root}/root/all_stays.csv')
    columns = ['subject_id', 'stay_id', 'intime', 'outtime']
    
    # only common subjects with both icu stay and an xray
    # Note that inner merge includes rows if a chest X-ray is associated with multiple stays
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns], how='inner', on='subject_id')
    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")

    cxr_merged_icustays.intime=pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime=pd.to_datetime(cxr_merged_icustays.outtime)
    
    cxr_merged_icustays['time_diff'] = cxr_merged_icustays.StudyDateTime-cxr_merged_icustays.intime
    cxr_merged_icustays['time_diff'] = cxr_merged_icustays['time_diff'].apply(lambda x: np.round(x.total_seconds()/60/60,3))

    cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays.outtime-cxr_merged_icustays.intime
    cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays['full_stay_time'].apply(lambda x: np.round(x.total_seconds()/60/60,3))

    return cxr_merged_icustays 

def get_recent_cxr(cxr_merged_icustays_AP):
    groups = cxr_merged_icustays_AP.groupby('stay_id')
    groups_selected = []
    for group in groups:
        # select the latest cxr for the icu stay
        selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
        groups_selected.append(selected)
    groups = pd.concat(groups_selected, ignore_index=True)
    groups['lower'] = 0
    groups['upper'] = groups.full_stay_time
    
    return groups

def get_all_cxr(cxr_merged_icustays_AP):
    groups = cxr_merged_icustays_AP.groupby('study_id').first()
    groups = groups.reset_index()
    groups = groups.groupby('study_id').first().sort_values(by=['stay_id','StudyDateTime'])
    groups = groups.reset_index()    
    groups['lower'] = 0
    groups['upper'] = groups.time_diff
    
    return groups 

def load_decompensation_meta(args):
    
    cxr_merged_icustays = get_all_metadata(args)
    train_listfile = pd.read_csv(f'mml-ssl/{args.task}/train_listfile.csv')
    val_listfile = pd.read_csv(f'mml-ssl/{args.task}/val_listfile.csv')
    test_listfile = pd.read_csv(f'mml-ssl/{args.task}/test_listfile.csv')
    
    # double check that this line does not cause issues
    listfile = train_listfile.append([val_listfile, test_listfile])
    listfile['subject_id'] = listfile['stay'].apply(lambda x: x.split("_")[0])
    listfile['subject_id'] = listfile['subject_id'].astype('int64')
    columns = ['subject_id', 'endtime']
    cxr_merged_icustays = cxr_merged_icustays.merge(listfile[columns], how='inner', on='subject_id')
    cxr_merged_icustays.endtime=pd.to_datetime(cxr_merged_icustays.endtime)
    cxr_merged_icustays_during = cxr_merged_icustays.loc[((cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                          (cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.endtime))]
    
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

    
def load_los_meta(args):

    cxr_merged_icustays = get_all_metadata(args)
    train_listfile = pd.read_csv(f'mml-ssl/{args.task}/train_listfile.csv')
    val_listfile = pd.read_csv(f'mml-ssl/{args.task}/val_listfile.csv')
    test_listfile = pd.read_csv(f'mml-ssl/{args.task}/test_listfile.csv')
    
    # double check that this line does not cause issues
    listfile = train_listfile.append([val_listfile, test_listfile])
    listfile['subject_id'] = listfile['stay'].apply(lambda x: x.split("_")[0])
    listfile['subject_id'] = listfile['subject_id'].astype('int64')
    columns = ['subject_id', 'endtime']
    cxr_merged_icustays = cxr_merged_icustays.merge(listfile[columns], how='inner', on='subject_id')
    cxr_merged_icustays.endtime=pd.to_datetime(cxr_merged_icustays.endtime)

    cxr_merged_icustays_during = cxr_merged_icustays.loc[((cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                            (cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.endtime))]

    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

def load_mortality_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                         ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups


def load_phenotyping_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.outtime
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                         ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

def load_readmission_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.outtime
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                         ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

def load_radiology_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.outtime
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                        ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_all_cxr(cxr_merged_icustays_AP)

    return groups

#NOTE: only if pretraining with large dataset, i.e. logic should be added to have a 'pretraining' task if args.dataset='all'
def load_pretraining_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    cxr_merged_icustays_AP = cxr_merged_icustays[cxr_merged_icustays['ViewPosition'] == 'AP']
    groups = cxr_merged_icustays_AP
    return groups

# check the task names to match the dictionary 
# check pre-training settings per task for supervised 
load_task_meta = {'decompensation':load_decompensation_meta,
                  'length-of-stay':load_los_meta,
                  'phenotyping':load_phenotyping_meta,
                  'in-hospital-mortality':load_mortality_meta,
                  'readmission':load_readmission_meta,
                  'radiology':load_radiology_meta,
                  'pretraining':load_pretraining_meta}


def count_labels(dataset):
    total_labels_count = 0
    for key, value in dataset.ehr_ds.data_map.items():
        # Access the labels for each key
        labels = value['labels']
        # Now you can work with the labels, for example, counting positive labels
        for label in labels:
            if label == 1:  # Assuming 1 indicates a positive label
                total_labels_count += 1 
    return total_labels_count




def get_final_meta(args):
    """
    Modified version that uses medmod_pairs.csv to filter available pairs,
    while still getting labels and metadata from the original sources
    """
    # Step 1: Load the medmod_pairs.csv to know what's actually available
    medmod_pairs = pd.read_csv(args.medmod_pairs_path)
    
    print("\n=== Medmod Pairs Debug ===")
    print(f"Columns in medmod_pairs: {list(medmod_pairs.columns)}")
    print(f"Shape: {medmod_pairs.shape}")
    print(f"First few rows:")
    print(medmod_pairs.head())
    
    # Extract dicom_id from image_path
    medmod_pairs['dicom_id'] = medmod_pairs['image_path'].apply(
        lambda x: x.split('/')[-1].replace('.jpg', '')
    )
    
    # Extract episode filename from episode_file path
    medmod_pairs['stay'] = medmod_pairs['episode_file'].apply(
        lambda x: '/'.join(x.split('/')[-3:])
    )
    
    # Convert time_diff_minutes to time_diff (in hours)
    medmod_pairs['time_diff'] = medmod_pairs['time_diff_minutes'] / 60.0
    
    # Step 2: Load cxr and ehr groups (original metadata with all columns)
    cxr_merged_icustays = load_task_meta[args.task](args)
    
    print("\n=== CXR Merged ICU Stays Debug ===")
    print(f"Columns in cxr_merged_icustays: {list(cxr_merged_icustays.columns)}")
    print(f"Shape: {cxr_merged_icustays.shape}")
    
    # Step 3: Add the labels from listfiles
    splits_labels_train = pd.read_csv(f'{args.ehr_data_root}/{args.task}/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{args.ehr_data_root}/{args.task}/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{args.ehr_data_root}/{args.task}/test_listfile.csv')
    
    # Merge to get labels
    train_meta_with_labels = cxr_merged_icustays.merge(splits_labels_train, how='inner', on='stay_id')
    val_meta_with_labels = cxr_merged_icustays.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = cxr_merged_icustays.merge(splits_labels_test, how='inner', on='stay_id')
    
    # Step 4: Get rid of chest X-rays that don't have radiology reports
    metadata = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-metadata.csv')
    labels = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-chexpert.csv')
    metadata_with_labels = metadata.merge(labels[['study_id']], how='inner', on='study_id').drop_duplicates(subset=['dicom_id'])
    
    train_meta_with_labels = train_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    val_meta_with_labels = val_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    test_meta_with_labels = test_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    
    # Step 5: CRITICAL - Filter by what's actually available in medmod_pairs.csv
    print(f"\nBefore filtering with medmod_pairs:")
    print(f"  Train: {len(train_meta_with_labels)}")
    print(f"  Val: {len(val_meta_with_labels)}")
    print(f"  Test: {len(test_meta_with_labels)}")
    
    # Create a set of available (dicom_id, stay) pairs from medmod_pairs
    available_pairs = set(zip(medmod_pairs['dicom_id'], medmod_pairs['stay']))
    
    # Filter train/val/test to only include available pairs
    def filter_available_pairs(meta_df, available_pairs):
        meta_df['pair_key'] = list(zip(meta_df['dicom_id'], meta_df['stay']))
        filtered_df = meta_df[meta_df['pair_key'].isin(available_pairs)].copy()
        filtered_df = filtered_df.drop(columns=['pair_key'])
        return filtered_df
    
    train_meta_with_labels = filter_available_pairs(train_meta_with_labels, available_pairs)
    val_meta_with_labels = filter_available_pairs(val_meta_with_labels, available_pairs)
    test_meta_with_labels = test_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    
    print(f"After filtering with medmod_pairs:")
    print(f"  Train: {len(train_meta_with_labels)}")
    print(f"  Val: {len(val_meta_with_labels)}")
    print(f"  Test: {len(test_meta_with_labels)}")
    
    # Step 6: Add columns from medmod_pairs (image_path, time_diff_minutes, time_diff, etc.)
    medmod_pairs_subset = medmod_pairs[['dicom_id', 'stay', 'image_path', 'time_diff_minutes', 'time_diff']].drop_duplicates()
    
    train_meta_with_labels = train_meta_with_labels.merge(
        medmod_pairs_subset, 
        how='inner', 
        on=['dicom_id', 'stay']
    )
    val_meta_with_labels = val_meta_with_labels.merge(
        medmod_pairs_subset, 
        how='inner', 
        on=['dicom_id', 'stay']
    )
    test_meta_with_labels = test_meta_with_labels.merge(
        medmod_pairs_subset, 
        how='inner', 
        on=['dicom_id', 'stay']
    )
    
    # Step 7: Ensure required columns exist
    # Add 'lower' and 'upper' if they don't exist
    for df in [train_meta_with_labels, val_meta_with_labels, test_meta_with_labels]:
        if 'lower' not in df.columns:
            df['lower'] = 0.0
        if 'upper' not in df.columns:
            # Use existing period_length or calculate from time_diff
            if 'period_length' in df.columns:
                df['upper'] = df['period_length']
            else:
                # Use absolute value of time_diff (in hours)
                df['upper'] = df['time_diff'].abs()
        if 'period_length' not in df.columns:
            df['period_length'] = df['upper']
    
    print(f"\n=== Final columns ===")
    print(f"Train columns: {list(train_meta_with_labels.columns)}")
    print(f"Sample of time_diff values (hours): {train_meta_with_labels['time_diff'].head()}")
    print(f"Sample of time_diff_minutes values: {train_meta_with_labels['time_diff_minutes'].head()}")
    
    return train_meta_with_labels, val_meta_with_labels, test_meta_with_labels

def load_cxr_ehr(args, 
                 ehr_train_ds, 
                 ehr_val_ds, 
                 cxr_train_ds, 
                 cxr_val_ds, 
                 ehr_test_ds, 
                 cxr_test_ds):

    # Load the medmod_pairs.csv instead of get_final_meta
    train_meta_with_labels, val_meta_with_labels, test_meta_with_labels = load_medmod_pairs(args)

    # Multimodal class
    train_ds = MIMIC_CXR_EHR(args, train_meta_with_labels, ehr_train_ds, cxr_train_ds)
    val_ds = MIMIC_CXR_EHR(args, val_meta_with_labels, ehr_val_ds, cxr_val_ds, split='val')
    test_ds = MIMIC_CXR_EHR(args, test_meta_with_labels, ehr_test_ds, cxr_test_ds, split='test')

    collate = my_collate_fusion
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) 
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate, drop_last=False) 
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    return train_dl, val_dl, test_dl

def load_medmod_pairs(args):
    """
    Load medmod_pairs.csv and split into train/val/test based on episode files
    """
    import pandas as pd
    
    # Load the pairs CSV
    pairs_df = pd.read_csv("./medmod_pairs.csv")  # You'll need to add this path to args
    
    # Extract dicom_id from image_path (last part before .jpg)
    pairs_df['dicom_id'] = pairs_df['image_path'].apply(
        lambda x: x.split('/')[-1].replace('.jpg', '')
    )
    
    # Determine split based on episode_file path (contains 'train', 'val', or 'test')
    def get_split(episode_file):
        if '/train/' in episode_file:
            return 'train'
        elif '/val/' in episode_file:
            return 'val'
        elif '/test/' in episode_file:
            return 'test'
        else:
            # Fallback: check if episode file exists in known splits
            return None
    
    pairs_df['split'] = pairs_df['episode_file'].apply(get_split)
    
    # Load CXR labels from metadata
    cxr_data_dir = args.cxr_data_root
    labels = pd.read_csv(f'{cxr_data_dir}/mimic-cxr-2.0.0-chexpert.csv')
    labels[R_CLASSES] = labels[R_CLASSES].fillna(0)
    labels = labels.replace(-1.0, 0.0)
    
    # Merge with CXR labels
    pairs_with_labels = pairs_df.merge(
        labels[R_CLASSES + ['study_id']], 
        how='inner', 
        on='study_id'
    )
    
    # Add columns needed for MIMIC_CXR_EHR dataset
    # Extract stay filename from episode_file
    pairs_with_labels['stay'] = pairs_with_labels['episode_file']
    
    # Calculate time boundaries for EHR data extraction
    # Convert time_diff from minutes to hours
    pairs_with_labels['time_diff_hours'] = pairs_with_labels['time_diff_minutes'] / 60.0
    
    # For paired data, we need to determine the time window
    # Assuming the CXR was taken at a specific time relative to admission
    # You may need to adjust this logic based on your specific requirements
    pairs_with_labels['period_length'] = pairs_with_labels['time_diff_hours'].abs()
    pairs_with_labels['upper'] = pairs_with_labels['period_length']
    pairs_with_labels['lower'] = 0.0  # Start from admission
    
    # Split into train/val/test
    train_meta = pairs_with_labels[pairs_with_labels['split'] == 'train'].reset_index(drop=True)
    val_meta = pairs_with_labels[pairs_with_labels['split'] == 'val'].reset_index(drop=True)
    test_meta = pairs_with_labels[pairs_with_labels['split'] == 'test'].reset_index(drop=True)
    
    return train_meta, val_meta, test_meta




def my_collate_fusion(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [x, img, targets_ehr, targets_cxr, seq_length, pairs]
    
