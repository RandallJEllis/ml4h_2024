import pandas as pd

def load_demographics(data_path):
    # import age, sex, education, site, assessment date
    df = pd.read_parquet(data_path +
                                   'demographics/demographics_df.parquet'
                                   )
    return df
    
def group_assessment_center(df, data_instance, assessment_centers_lookup):
    # convert region
    assessment_centers = pd.DataFrame(df.loc[:, f'54-{data_instance}.0'])
    assessment_centers = assessment_centers.merge(assessment_centers_lookup, left_on=f'54-{data_instance}.0', right_on='coding')

    assessment_centers['region_label'] = ''
    assessment_centers.loc[assessment_centers.meaning.isin(['Barts', 'Hounslow', 'Croydon']), 'region_label'] = 'London'
    assessment_centers.loc[assessment_centers.meaning.isin(['Wrexham', 'Swansea', 'Cardiff']), 'region_label'] = 'Wales'
    assessment_centers.loc[assessment_centers.meaning.isin(['Cheadle (imaging)', 'Cheadle (revisit)', 'Stockport',
                                                            'Stockport (pilot)', 'Manchester', 'Liverpool','Bury']),
                                                             'region_label'] = 'North-West'
    assessment_centers.loc[assessment_centers.meaning.isin(['Newcastle', 'Newcastle (imaging)', 'Middlesborough']), 'region_label'] = 'North-East'
    assessment_centers.loc[assessment_centers.meaning.isin(['Leeds','Sheffield']), 'region_label'] = 'Yorkshire and Humber'
    assessment_centers.loc[assessment_centers.meaning.isin(['Stoke','Birmingham']), 'region_label'] = 'West Midlands'
    assessment_centers.loc[assessment_centers.meaning.isin(['Nottingham']), 'region_label'] = 'East Midlands'
    assessment_centers.loc[assessment_centers.meaning.isin(['Oxford', 'Reading', 'Reading (imaging)']), 'region_label'] = 'South-East'
    assessment_centers.loc[assessment_centers.meaning.isin(['Bristol', 'Bristol (imaging)']), 'region_label'] = 'South-West'
    assessment_centers.loc[assessment_centers.meaning.isin(['Glasgow', 'Edinburgh']), 'region_label'] = 'Scotland'

    region_indices = assessment_centers.groupby('region_label').groups

    return region_indices

def get_last_completed_education(df, instance):
    # education columns have array indices 0-5 for each instance
    # for each patient, take the max value of the Instance 0 columns
    # to represent max education completed
    educ_cols = df.columns[df.columns.str.startswith(f'6138-{instance}')]
    max_educ = df.loc[:, educ_cols].max(axis=1)
    df['max_educ_complete'] = max_educ
    return df

