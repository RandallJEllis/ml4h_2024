import pandas as pd
import numpy as np
import df_utils


def remove_pre_instance_dementia(df, instance, dementia_df):
    'Remove dementia before the UKB visit at a specified instance and create a label column for the cases'

    disease_source = df_utils.pull_columns_by_prefix(dementia_df, ['eid', '42019', '42021', '42023', '42025'])
    values_set = [1, 2, 11, 12, 21, 22]
    keep_disease_source = df_utils.pull_rows_with_values(disease_source, values_set)

    icd_source = df_utils.pull_columns_by_prefix(dementia_df, ['eid', '131037', '130837', '130839', '130841', '130843'])
    values_set = [20, 21, 30, 31, 40, 41, 51]
    keep_icd_source = df_utils.pull_rows_with_values(icd_source, values_set)

    both_eid = set(keep_disease_source.eid).union(set(keep_icd_source.eid))
    remove = (set(disease_source.eid).union(set(icd_source.eid))).difference(both_eid)

    date_df = df_utils.pull_columns_by_prefix(dementia_df, ['eid', '42018', '42020', '42022', '42024', '131036',
                                '130836', '130838', '130840', '130842'])
    date_df = date_df[date_df.eid.isin(both_eid)]

    exclude_df = df_utils.pull_columns_by_prefix(dementia_df, ['eid', '42019', '42021', '42023', '42025',
                    '131037', '130837', '130839', '130841', '130843',
                    '42018', '42020', '42022', '42024', '131036',
                                '130836', '130838', '130840', '130842'])

    # remove patients diagnosed with dementia before proteomics
    for col in date_df.columns[1:].tolist():
        date_df[col] = pd.to_datetime(date_df.loc[:, col])
    date_df['first_dx'] = date_df.iloc[:, 1:].min(axis=1)

    df[f'53-{instance}.0'] = pd.to_datetime(df[f'53-{instance}.0'])
    cases = df.merge(date_df, on='eid')

    cases['date_diff'] = cases.first_dx - cases[f'53-{instance}.0']
    cases = cases[cases.date_diff > pd.Timedelta(0)]
    cases = cases[cases.eid.isin(both_eid)]

    controls = df[~df.eid.isin(exclude_df.eid)]
    cases = df[df.eid.isin(cases.eid)]

    df = pd.concat([controls, cases])
    df['label'] = df['eid'].isin(cases.eid).astype(int)

    return df
    
    
def apoe_alleles(df, alleles, genotype=False):
    '''
    Apolipoprotein E (APOE) genotype has three major alleles
    (epsilon 2, epsilon 3, epsilon 4):
    epsilon2 - rs429358-T, rs7412-T
    epsilon3 - rs429358-T, rs7412-C
    epsilon4 - rs429358-C, rs7412-C

    These two single nucleotide polymorphisms (SNPs; rs429358, rs7412)
    were extracted from the UK Biobank genomics data using Plink2.

    The relevant columns of this output are:
    IID - The equivalent of eid in our other files
    rs429358_C - 1 means they have a C at the genomic locus;
    0 means they have a T
    rs7412_T - 1 means they have a T at the genomic locus;
    0 means they have a C

    If genotype is False, just count alleles for rs429358
    If genotype is True, encode these polymorphisms based on both allele columns.

    '''

    alleles['apoe_polymorphism'] = np.nan

    if genotype == True:
        alleles.loc[(alleles.rs429358_C == 0) &
                    (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e3/e3'
        alleles.loc[(alleles.rs429358_C == 1) &
                    (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e3/e4'
        alleles.loc[(alleles.rs429358_C == 0) &
                    (alleles.rs7412_T == 1), 'apoe_polymorphism'] = 'e2/e3'
        alleles.loc[(alleles.rs429358_C == 1) &
                    (alleles.rs7412_T == 1), 'apoe_polymorphism'] = 'e2/e4'
        alleles.loc[(alleles.rs429358_C == 2) &
                    (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e4/e4'
        alleles.loc[(alleles.rs429358_C == 0) &
                    (alleles.rs7412_T == 2), 'apoe_polymorphism'] = 'e2/e2'

    else:
        alleles.loc[(alleles.rs429358_C == 0), 'apoe_polymorphism'] = 0
        alleles.loc[(alleles.rs429358_C == 1), 'apoe_polymorphism'] = 1
        alleles.loc[(alleles.rs429358_C == 2), 'apoe_polymorphism'] = 2

    df = df.merge(alleles[['IID', 'apoe_polymorphism']], left_on='eid',
                    right_on='IID', how='left')

    return df

