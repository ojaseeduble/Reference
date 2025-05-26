# FINAL LTV TOOL (v4): All methods with fallback logic + original input order
import pandas as pd
import networkx as nx

def load_clean_data(filepath):
    rows_to_drop = list(range(0, 4)) + [5, 6]
    df_loans = pd.read_excel(filepath, sheet_name='2.-Loan', header=None)
    df_collateral = pd.read_excel(filepath, sheet_name='4.-Loan & Collateral', header=None)

    df_loans = df_loans.drop(index=rows_to_drop).reset_index(drop=True)
    df_collateral = df_collateral.drop(index=rows_to_drop).reset_index(drop=True)

    df_loans.columns = df_loans.iloc[0]
    df_loans = df_loans.drop(index=0).reset_index(drop=True)
    df_collateral.columns = df_collateral.iloc[0]
    df_collateral = df_collateral.drop(index=0).reset_index(drop=True)

    df_loans = df_loans[['Loan reference', 'Total outstanding debt as of 29.02.2024', 'Borrower reference']]
    df_collateral = df_collateral[['Loan reference', 'Collateral unit reference', 'Plot',
                                   'Gross Appraisal Value', 'Priority Ranking', 'Collateral type', 'Province']]
    df_collateral = df_collateral[df_collateral['Gross Appraisal Value'] != 'n.a.']

    df_collateral['Asset ID'] = df_collateral['Collateral unit reference'].astype(str) + '__' + df_collateral['Plot'].astype(str)
    df_loans['Total outstanding debt as of 29.02.2024'] = pd.to_numeric(df_loans['Total outstanding debt as of 29.02.2024'], errors='coerce')
    df_collateral['Gross Appraisal Value'] = pd.to_numeric(df_collateral['Gross Appraisal Value'], errors='coerce')

    return df_loans, df_collateral


# --- Method 1: Loan-Level Conservative & Aggressive LTV (Updated Full Logic) ---
def calculate_loan_level_ltv_with_fallback(df_loans, df_collateral):
    df_loans = df_loans[['Loan reference', 'Original loan balance', 'Borrower reference']]
    df_collateral = df_collateral[['Loan reference', 'Collateral unit reference', 'Plot',
                                   'Original Gross Appraisal Value', 'Priority Ranking',
                                   'Collateral type', 'Province', 'Date of original valuation']]

    all_loan_refs = set(df_collateral['Loan reference'])
    df_collateral = df_collateral[df_collateral['Original Gross Appraisal Value'] != 'n.a.']
    remaining_loan_refs = set(df_collateral['Loan reference'])
    dropped_loan_refs = all_loan_refs - remaining_loan_refs
    print(f"Dropped {len(dropped_loan_refs)} loan references due to 'n.a.' values:")
    print(dropped_loan_refs)

    df_collateral['Asset ID'] = df_collateral['Collateral unit reference'].astype(str) + '__' + df_collateral['Plot'].astype(str)
    df_loans['Original loan balance'] = pd.to_numeric(df_loans['Original loan balance'], errors='coerce')
    df_collateral['Original Gross Appraisal Value'] = pd.to_numeric(df_collateral['Original Gross Appraisal Value'], errors='coerce')

    df_merged = pd.merge(df_collateral, df_loans, on='Loan reference', how='left')

    loan_province_map = df_merged.groupby('Loan reference')['Province'].first().to_dict()
    loan_valuation_date_map = df_merged.groupby('Loan reference')['Date of original valuation'].first().to_dict()
    loan_order = {loan_ref: idx for idx, loan_ref in enumerate(df_loans['Loan reference'])}
    loan_lien_positions = df_merged.groupby('Loan reference')['Priority Ranking'].unique().apply(list).to_dict()

    df_lien1 = df_merged[df_merged['Priority Ranking'] == 'Lien 1'].copy()
    df_lien_gt1 = df_merged[df_merged['Priority Ranking'] != 'Lien 1'].copy()

    df_lien1['LTV_approx'] = df_lien1['Original loan balance'] / df_lien1['Original Gross Appraisal Value']
    lien1_debt_map = df_lien1.groupby('Asset ID')['Original loan balance'].sum().reset_index()
    lien1_debt_map.columns = ['Asset ID', 'Lien 1 Debt']
    asset_to_lien1_debt = lien1_debt_map.set_index('Asset ID')['Lien 1 Debt'].to_dict()

    df_lien1 = df_lien1.merge(lien1_debt_map, on='Asset ID', how='left')
    df_lien1['Allocated AV Conservative'] = df_lien1['Original Gross Appraisal Value'] * df_lien1['Original loan balance'] / df_lien1['Lien 1 Debt']
    df_lien1['Allocated AV Aggressive'] = df_lien1['Original Gross Appraisal Value']

    asset_to_lien1_loans = df_lien1[['Asset ID', 'Loan reference']].drop_duplicates().groupby('Asset ID')['Loan reference'].apply(list).to_dict()
    loan_to_debt = dict(zip(df_lien1['Loan reference'], df_lien1['Original loan balance']))
    loan_to_lien1_av = df_lien1.groupby('Loan reference')['Allocated AV Conservative'].sum().to_dict()

    df_lien1_result = df_lien1.groupby('Loan reference').agg({
        'Borrower reference': 'first',
        'Original loan balance': 'first',
        'Allocated AV Conservative': 'sum',
        'Allocated AV Aggressive': 'sum'
    }).reset_index()
    df_lien1_result['Conservative LTV (%)'] = df_lien1_result['Original loan balance'] / df_lien1_result['Allocated AV Conservative'] * 100
    df_lien1_result['Aggressive LTV (%)'] = df_lien1_result['Original loan balance'] / df_lien1_result['Allocated AV Aggressive'] * 100
    df_lien1_result['Used Fallback'] = False

    df_fallback_base = df_lien1_result.merge(
        df_merged[['Loan reference', 'Province', 'Collateral type']].drop_duplicates(),
        on='Loan reference', how='left')
    fallback_ref = df_fallback_base.groupby(['Province', 'Collateral type']).apply(
        lambda g: (g['Conservative LTV (%)'] * g['Original loan balance']).sum() / g['Original loan balance'].sum()
    ).reset_index(name='Weighted Avg Lien1 LTV')

    loan_group_list = []
    mixed_lien_loans = [loan for loan, positions in loan_lien_positions.items()
                        if 'Lien 1' in positions and any(pos != 'Lien 1' for pos in positions)]

    for loan_ref, group in df_lien_gt1.groupby('Loan reference'):
        total_av = group['Original Gross Appraisal Value'].sum()
        borrower = group['Borrower reference'].iloc[0]
        loan_outstanding = group['Original loan balance'].iloc[0]

        seen_lien1_loans = set()
        lien1_total_debt = 0
        fallback_used = False

        for _, row in group.iterrows():
            asset = row['Asset ID']
            province = row['Province']
            ctype = row['Collateral type']
            gross_av = row['Original Gross Appraisal Value']
            lien1_loans_on_asset = asset_to_lien1_loans.get(asset, [])
            if lien1_loans_on_asset:
                for lien1_loan in lien1_loans_on_asset:
                    if lien1_loan not in seen_lien1_loans and lien1_loan in loan_to_debt:
                        lien1_total_debt += loan_to_debt[lien1_loan]
                        seen_lien1_loans.add(lien1_loan)
            else:
                fallback_row = fallback_ref[(fallback_ref['Province'] == province) & (fallback_ref['Collateral type'] == ctype)]
                if not fallback_row.empty:
                    fallback_ltv = fallback_row['Weighted Avg Lien1 LTV'].values[0] / 100
                    lien1_total_debt += fallback_ltv * gross_av
                    fallback_used = True

        allocated_av_gt1 = total_av - lien1_total_debt if total_av > lien1_total_debt else 0
        additional_lien1_av = loan_to_lien1_av.get(loan_ref, 0) if loan_ref in mixed_lien_loans else 0
        total_allocated_av = allocated_av_gt1 + additional_lien1_av
        conservative_ltv = loan_outstanding / total_allocated_av * 100 if total_allocated_av > 0 else 0
        aggressive_ltv = loan_outstanding / total_av * 100 if total_av > 0 else 0

        loan_group_list.append({
            'Loan reference': str(loan_ref),
            'Original Gross Appraisal Value': float(total_av),
            'Estimated Lien 1 Debt': float(lien1_total_debt),
            'Original loan balance': float(loan_outstanding),
            'Borrower reference': str(borrower),
            'Allocated AV Conservative': float(total_allocated_av),
            'Allocated AV Aggressive': float(total_av),
            'Conservative LTV (%)': conservative_ltv,
            'Aggressive LTV (%)': aggressive_ltv,
            'Used Fallback': fallback_used
        })

    df_gt1_result = pd.DataFrame(loan_group_list)
    df_combined = pd.concat([df_lien1_result, df_gt1_result], axis=0)

    df_combined = df_combined.groupby(['Borrower reference', 'Loan reference'], as_index=False).agg({
        'Original loan balance': 'first',
        'Allocated AV Conservative': 'sum',
        'Allocated AV Aggressive': 'sum',
        'Used Fallback': 'max'
    })
    df_combined['Conservative LTV (%)'] = df_combined['Original loan balance'] / df_combined['Allocated AV Conservative']
    df_combined['Aggressive LTV (%)'] = df_combined['Original loan balance'] / df_combined['Allocated AV Aggressive']
    df_combined['Province'] = df_combined['Loan reference'].map(loan_province_map)
    df_combined['Date of original valuation'] = df_combined['Loan reference'].map(loan_valuation_date_map)
    df_combined['original_order'] = df_combined['Loan reference'].map(loan_order)
    df_combined = df_combined.sort_values('original_order').drop(columns=['original_order'])

    return df_combined


# --- Method 2: Borrower-Level LTV with Fallback + Order ---
def calculate_borrower_based_ltv_with_fallback(df_loans, df_collateral):
    loan_order = {loan_ref: idx for idx, loan_ref in enumerate(df_loans['Loan reference'])}
    loan_to_borrower = df_loans.set_index('Loan reference')['Borrower reference'].to_dict()
    df_collateral['Borrower reference'] = df_collateral['Loan reference'].map(loan_to_borrower)

    df_lien1 = df_collateral[df_collateral['Priority Ranking'] == 'Lien 1'].copy()
    df_lien_gt1 = df_collateral[df_collateral['Priority Ranking'] != 'Lien 1'].copy()

    df_lien1_merged = pd.merge(df_lien1, df_loans, on='Loan reference', how='left')
    fallback_lien1 = df_lien1_merged[['Loan reference', 'Borrower reference', 'Gross Appraisal Value', 'Total outstanding debt as of 29.02.2024']]
    fallback_lien1 = fallback_lien1.groupby('Loan reference').agg({
        'Borrower reference': 'first',
        'Gross Appraisal Value': 'sum',
        'Total outstanding debt as of 29.02.2024': 'sum'
    }).reset_index()
    fallback_lien1['Conservative LTV (%)'] = fallback_lien1['Total outstanding debt as of 29.02.2024'] / fallback_lien1['Gross Appraisal Value'] * 100

    fallback_base = pd.merge(fallback_lien1, df_collateral[['Loan reference', 'Province', 'Collateral type']].drop_duplicates(), on='Loan reference', how='left')
    fallback_ref = fallback_base.groupby(['Province', 'Collateral type']).apply(
        lambda g: (g['Conservative LTV (%)'] * g['Total outstanding debt as of 29.02.2024']).sum() /
                  g['Total outstanding debt as of 29.02.2024'].sum()
    ).reset_index(name='Weighted Avg Lien1 LTV')

    borrower_records = []
    for _, row in df_loans.iterrows():
        loan = row['Loan reference']
        borrower = row['Borrower reference']
        loan_debt = row['Total outstanding debt as of 29.02.2024']

        asset_rows = df_collateral[df_collateral['Loan reference'] == loan]
        lien1_assets = asset_rows[asset_rows['Priority Ranking'] == 'Lien 1']
        gt1_assets = asset_rows[asset_rows['Priority Ranking'] != 'Lien 1']

        lien1_av = lien1_assets['Gross Appraisal Value'].sum()
        gt1_av = gt1_assets['Gross Appraisal Value'].sum()

        estimated_lien1_debt = 0
        fallback_used = False
        for _, asset_row in gt1_assets.iterrows():
            province = asset_row['Province']
            ctype = asset_row['Collateral type']
            gross_av = asset_row['Gross Appraisal Value']
            fallback_row = fallback_ref[(fallback_ref['Province'] == province) & (fallback_ref['Collateral type'] == ctype)]
            if not fallback_row.empty:
                fallback_ltv = fallback_row['Weighted Avg Lien1 LTV'].values[0] / 100
                estimated_lien1_debt += fallback_ltv * gross_av
                fallback_used = True

        total_av = lien1_av + gt1_av
        total_lien1_debt = estimated_lien1_debt if lien1_av == 0 else loan_debt if gt1_av == 0 else estimated_lien1_debt
        adjusted_av_gt1 = gt1_av - total_lien1_debt if gt1_av > total_lien1_debt else 0
        ltv_total = (loan_debt / total_av * 100) if total_av else 0
        ltv_lien_gt1 = (loan_debt / adjusted_av_gt1 * 100) if adjusted_av_gt1 else 0

        borrower_records.append({
            'Loan reference': loan,
            'Borrower reference': borrower,
            'Total outstanding debt as of 29.02.2024': loan_debt,
            'Lien 1 Appraisal Value': lien1_av,
            'Lien > 1 Appraisal Value': gt1_av,
            'Estimated Lien 1 Debt': total_lien1_debt,
            'Adjusted Lien > 1 Value': adjusted_av_gt1,
            'Total LTV (%)': ltv_total,
            'Lien > 1 LTV (%)': ltv_lien_gt1,
            'Used Fallback': fallback_used
        })

    df_result = pd.DataFrame(borrower_records)
    df_result['original_order'] = df_result['Loan reference'].map(loan_order)
    return df_result.sort_values('original_order').drop(columns='original_order')


# --- Method 3: Loan-Level LTV with Fallback (same as Method 1) ---
def calculate_loan_level_ltv_with_fallback(df_loans, df_collateral):
    df_loans = df_loans[['Loan reference', 'Original loan balance', 'Borrower reference']]
    df_collateral = df_collateral[['Loan reference', 'Collateral unit reference', 'Plot',
                                   'Original Gross Appraisal Value', 'Priority Ranking',
                                   'Collateral type', 'Province', 'Date of original valuation']]

    all_loan_refs = set(df_collateral['Loan reference'])
    df_collateral = df_collateral[df_collateral['Original Gross Appraisal Value'] != 'n.a.']
    remaining_loan_refs = set(df_collateral['Loan reference'])
    dropped_loan_refs = all_loan_refs - remaining_loan_refs
    print(f"Dropped {len(dropped_loan_refs)} loan references due to 'n.a.' values:")
    print(dropped_loan_refs)

    df_collateral['Asset ID'] = df_collateral['Collateral unit reference'].astype(str) + '__' + df_collateral['Plot'].astype(str)
    df_loans['Original loan balance'] = pd.to_numeric(df_loans['Original loan balance'], errors='coerce')
    df_collateral['Original Gross Appraisal Value'] = pd.to_numeric(df_collateral['Original Gross Appraisal Value'], errors='coerce')

    df_merged = pd.merge(df_collateral, df_loans, on='Loan reference', how='left')

    loan_province_map = df_merged.groupby('Loan reference')['Province'].first().to_dict()
    loan_valuation_date_map = df_merged.groupby('Loan reference')['Date of original valuation'].first().to_dict()
    loan_order = {loan_ref: idx for idx, loan_ref in enumerate(df_loans['Loan reference'])}
    loan_lien_positions = df_merged.groupby('Loan reference')['Priority Ranking'].unique().apply(list).to_dict()

    df_lien1 = df_merged[df_merged['Priority Ranking'] == 'Lien 1'].copy()
    df_lien_gt1 = df_merged[df_merged['Priority Ranking'] != 'Lien 1'].copy()

    df_lien1['LTV_approx'] = df_lien1['Original loan balance'] / df_lien1['Original Gross Appraisal Value']
    lien1_debt_map = df_lien1.groupby('Asset ID')['Original loan balance'].sum().reset_index()
    lien1_debt_map.columns = ['Asset ID', 'Lien 1 Debt']
    asset_to_lien1_debt = lien1_debt_map.set_index('Asset ID')['Lien 1 Debt'].to_dict()

    df_lien1 = df_lien1.merge(lien1_debt_map, on='Asset ID', how='left')
    df_lien1['Allocated AV Conservative'] = df_lien1['Original Gross Appraisal Value'] * df_lien1['Original loan balance'] / df_lien1['Lien 1 Debt']
    df_lien1['Allocated AV Aggressive'] = df_lien1['Original Gross Appraisal Value']

    asset_to_lien1_loans = df_lien1[['Asset ID', 'Loan reference']].drop_duplicates().groupby('Asset ID')['Loan reference'].apply(list).to_dict()
    loan_to_debt = dict(zip(df_lien1['Loan reference'], df_lien1['Original loan balance']))
    loan_to_lien1_av = df_lien1.groupby('Loan reference')['Allocated AV Conservative'].sum().to_dict()

    df_lien1_result = df_lien1.groupby('Loan reference').agg({
        'Borrower reference': 'first',
        'Original loan balance': 'first',
        'Allocated AV Conservative': 'sum',
        'Allocated AV Aggressive': 'sum'
    }).reset_index()
    df_lien1_result['Conservative LTV (%)'] = df_lien1_result['Original loan balance'] / df_lien1_result['Allocated AV Conservative'] * 100
    df_lien1_result['Aggressive LTV (%)'] = df_lien1_result['Original loan balance'] / df_lien1_result['Allocated AV Aggressive'] * 100
    df_lien1_result['Used Fallback'] = False

    df_fallback_base = df_lien1_result.merge(
        df_merged[['Loan reference', 'Province', 'Collateral type']].drop_duplicates(),
        on='Loan reference', how='left')
    fallback_ref = df_fallback_base.groupby(['Province', 'Collateral type']).apply(
        lambda g: (g['Conservative LTV (%)'] * g['Original loan balance']).sum() / g['Original loan balance'].sum()
    ).reset_index(name='Weighted Avg Lien1 LTV')

    loan_group_list = []
    mixed_lien_loans = [loan for loan, positions in loan_lien_positions.items()
                        if 'Lien 1' in positions and any(pos != 'Lien 1' for pos in positions)]

    for loan_ref, group in df_lien_gt1.groupby('Loan reference'):
        total_av = group['Original Gross Appraisal Value'].sum()
        borrower = group['Borrower reference'].iloc[0]
        loan_outstanding = group['Original loan balance'].iloc[0]

        seen_lien1_loans = set()
        lien1_total_debt = 0
        fallback_used = False

        for _, row in group.iterrows():
            asset = row['Asset ID']
            province = row['Province']
            ctype = row['Collateral type']
            gross_av = row['Original Gross Appraisal Value']
            lien1_loans_on_asset = asset_to_lien1_loans.get(asset, [])
            if lien1_loans_on_asset:
                for lien1_loan in lien1_loans_on_asset:
                    if lien1_loan not in seen_lien1_loans and lien1_loan in loan_to_debt:
                        lien1_total_debt += loan_to_debt[lien1_loan]
                        seen_lien1_loans.add(lien1_loan)
            else:
                fallback_row = fallback_ref[(fallback_ref['Province'] == province) & (fallback_ref['Collateral type'] == ctype)]
                if not fallback_row.empty:
                    fallback_ltv = fallback_row['Weighted Avg Lien1 LTV'].values[0] / 100
                    lien1_total_debt += fallback_ltv * gross_av
                    fallback_used = True

        allocated_av_gt1 = total_av - lien1_total_debt if total_av > lien1_total_debt else 0
        additional_lien1_av = loan_to_lien1_av.get(loan_ref, 0) if loan_ref in mixed_lien_loans else 0
        total_allocated_av = allocated_av_gt1 + additional_lien1_av
        conservative_ltv = loan_outstanding / total_allocated_av * 100 if total_allocated_av > 0 else 0
        aggressive_ltv = loan_outstanding / total_av * 100 if total_av > 0 else 0

        loan_group_list.append({
            'Loan reference': str(loan_ref),
            'Original Gross Appraisal Value': float(total_av),
            'Estimated Lien 1 Debt': float(lien1_total_debt),
            'Original loan balance': float(loan_outstanding),
            'Borrower reference': str(borrower),
            'Allocated AV Conservative': float(total_allocated_av),
            'Allocated AV Aggressive': float(total_av),
            'Conservative LTV (%)': conservative_ltv,
            'Aggressive LTV (%)': aggressive_ltv,
            'Used Fallback': fallback_used
        })

    df_gt1_result = pd.DataFrame(loan_group_list)
    df_combined = pd.concat([df_lien1_result, df_gt1_result], axis=0)

    df_combined = df_combined.groupby(['Borrower reference', 'Loan reference'], as_index=False).agg({
        'Original loan balance': 'first',
        'Allocated AV Conservative': 'sum',
        'Allocated AV Aggressive': 'sum',
        'Used Fallback': 'max'
    })
    df_combined['Conservative LTV (%)'] = df_combined['Original loan balance'] / df_combined['Allocated AV Conservative']
    df_combined['Aggressive LTV (%)'] = df_combined['Original loan balance'] / df_combined['Allocated AV Aggressive']
    df_combined['Province'] = df_combined['Loan reference'].map(loan_province_map)
    df_combined['Date of original valuation'] = df_combined['Loan reference'].map(loan_valuation_date_map)
    df_combined['original_order'] = df_combined['Loan reference'].map(loan_order)
    df_combined = df_combined.sort_values('original_order').drop(columns=['original_order'])

    return df_combined


# --- Method 4: Component-Based LTV with Fallback + Order ---
def calculate_comp_based_ltv_expanded(df_loans, df_collateral):
    loan_order = {loan_ref: idx for idx, loan_ref in enumerate(df_loans['Loan reference'])}
    df_collateral['Asset ID'] = df_collateral['Collateral unit reference'].astype(str) + '__' + df_collateral['Plot'].astype(str)
    df_merged = pd.merge(df_collateral, df_loans, on='Loan reference', how='left')

    df_lien1 = df_merged[df_merged['Priority Ranking'] == 'Lien 1'].copy()
    fallback_lien1 = df_lien1[['Loan reference', 'Borrower reference', 'Gross Appraisal Value', 'Total outstanding debt as of 29.02.2024']]
    fallback_lien1 = fallback_lien1.groupby('Loan reference').agg({
        'Borrower reference': 'first',
        'Gross Appraisal Value': 'sum',
        'Total outstanding debt as of 29.02.2024': 'sum'
    }).reset_index()
    fallback_lien1['Conservative LTV (%)'] = fallback_lien1['Total outstanding debt as of 29.02.2024'] / fallback_lien1['Gross Appraisal Value'] * 100

    fallback_base = pd.merge(fallback_lien1, df_collateral[['Loan reference', 'Province', 'Collateral type']].drop_duplicates(), on='Loan reference', how='left')
    fallback_ref = fallback_base.groupby(['Province', 'Collateral type']).apply(
        lambda g: (g['Conservative LTV (%)'] * g['Total outstanding debt as of 29.02.2024']).sum() /
                  g['Total outstanding debt as of 29.02.2024'].sum()
    ).reset_index(name='Weighted Avg Lien1 LTV')

    G = nx.Graph()
    for _, row in df_merged.iterrows():
        G.add_edge('L_' + str(row['Loan reference']), 'A_' + str(row['Asset ID']))
    components = list(nx.connected_components(G))
    expanded_records = []

    for comp in components:
        loans = [node[2:] for node in comp if node.startswith('L_')]
        assets = [node[2:] for node in comp if node.startswith('A_')]
        comp_links = df_merged[df_merged['Loan reference'].isin(loans) & df_merged['Asset ID'].isin(assets)]

        total_gav = comp_links[['Asset ID', 'Gross Appraisal Value']].drop_duplicates()['Gross Appraisal Value'].sum()
        lien1_loans = comp_links[comp_links['Priority Ranking'] == 'Lien 1'].copy()
        gt1_loans = comp_links[comp_links['Priority Ranking'] != 'Lien 1'].copy()
        total_lien1_debt = lien1_loans[['Loan reference', 'Total outstanding debt as of 29.02.2024']].drop_duplicates()['Total outstanding debt as of 29.02.2024'].sum()
        total_lien_gt1_debt = gt1_loans[['Loan reference', 'Total outstanding debt as of 29.02.2024']].drop_duplicates()['Total outstanding debt as of 29.02.2024'].sum()

        # Apply fallback logic for lien 1 where missing
        fallback_used = False
        for _, row in gt1_loans.iterrows():
            province = row['Province']
            ctype = row['Collateral type']
            asset = row['Asset ID']
            if asset not in lien1_loans['Asset ID'].values:
                fallback_row = fallback_ref[(fallback_ref['Province'] == province) & (fallback_ref['Collateral type'] == ctype)]
                if not fallback_row.empty:
                    fallback_ltv = fallback_row['Weighted Avg Lien1 LTV'].values[0] / 100
                    total_lien1_debt += fallback_ltv * row['Gross Appraisal Value']
                    fallback_used = True

        adjusted_av_gt1 = total_gav - total_lien1_debt if total_gav > total_lien1_debt else 0
        ltv_total = (total_lien1_debt + total_lien_gt1_debt) / total_gav * 100 if total_gav else 0
        ltv_lien1 = total_lien1_debt / total_gav * 100 if total_gav else 0
        ltv_lien_gt1 = total_lien_gt1_debt / adjusted_av_gt1 * 100 if adjusted_av_gt1 else 0

        for loan in loans:
            expanded_records.append({
                'Component Loan': loan,
                'Component Assets': ', '.join(assets),
                'Component Total outstanding debt as of 29.02.2024': total_lien1_debt + total_lien_gt1_debt,
                'Total Lien 1 Debt': total_lien1_debt,
                'Total Lien > 1 Debt': total_lien_gt1_debt,
                'Lien 1 LTV %': ltv_lien1,
                'Lien > 1 LTV %': ltv_lien_gt1,
                'Component Gross AV': total_gav,
                'Component LTV (%)': ltv_total,
                'Used Fallback': fallback_used,
                'original_order': loan_order.get(loan, -1)
            })

    df_result = pd.DataFrame(expanded_records)
    return df_result.sort_values('original_order').drop(columns='original_order')


def main():
    print("Select LTV Calculation Method:")
    print("1 - Loan-Level Conservative & Aggressive LTV (Full Logic with Sorting, Date, Province)")
    print("2 - Borrower-Level LTV with Fallback")
    print("3 - Loan-Level LTV with Fallback")
    print("4 - Component-Based LTV with Fallback (Network Graph)")

    choice = input("Enter the number corresponding to your choice: ")
    filepath = input("Enter the full path to the Excel data file: ")

    df_loans, df_collateral = load_clean_data(filepath)

    if choice == '1':
        result = calculate_loan_level_ltv_with_fallback(df_loans, df_collateral)
        result.to_excel("ltv_loan_level_full.xlsx", index=False)
        print("Loan-Level LTV (Full Logic) exported to 'ltv_loan_level_full.xlsx'")
    elif choice == '2':
        result = calculate_borrower_based_ltv_with_fallback(df_loans, df_collateral)
        result.to_excel("ltv_borrower_with_fallback.xlsx", index=False)
        print("Borrower-Level LTV with Fallback exported to 'ltv_borrower_with_fallback.xlsx'")
    elif choice == '3':
        result = calculate_loan_level_ltv_with_fallback(df_loans, df_collateral)
        result.to_excel("ltv_loan_with_fallback.xlsx", index=False)
        print("Loan-Level LTV with Fallback exported to 'ltv_loan_with_fallback.xlsx'")
    elif choice == '4':
        result = calculate_comp_based_ltv_expanded(df_loans, df_collateral)
        result.to_excel("ltv_component_based.xlsx", index=False)
        print("Component-Based LTV exported to 'ltv_component_based.xlsx'")
    else:
        print("Invalid choice. Please enter a number from 1 to 4.")

# Uncomment below to run as script
# if __name__ == "__main__":
#     main()
