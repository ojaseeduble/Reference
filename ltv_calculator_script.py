
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

def calculate_loan_level_ltv(df_loans, df_collateral):
    df_merged = pd.merge(df_collateral, df_loans, on='Loan reference', how='left')
    loan_lien_positions = df_merged.groupby('Loan reference')['Priority Ranking'].unique().apply(list).to_dict()
    df_lien1 = df_merged[df_merged['Priority Ranking'] == 'Lien 1'].copy()

    df_lien1['LTV_approx'] = df_lien1['Total outstanding debt as of 29.02.2024'] / df_lien1['Gross Appraisal Value']
    lien1_debt_map = df_lien1.groupby('Asset ID')['Total outstanding debt as of 29.02.2024'].sum().reset_index()
    lien1_debt_map.columns = ['Asset ID', 'Lien 1 Debt']
    df_lien1 = df_lien1.merge(lien1_debt_map, on='Asset ID', how='left')
    df_lien1['Allocated AV Conservative'] = df_lien1['Gross Appraisal Value'] * df_lien1['Total outstanding debt as of 29.02.2024'] / df_lien1['Lien 1 Debt']
    df_lien1['Allocated AV Aggressive'] = df_lien1['Gross Appraisal Value']

    df_lien1_result = df_lien1.groupby('Loan reference').agg({
        'Borrower reference': 'first',
        'Total outstanding debt as of 29.02.2024': 'first',
        'Allocated AV Conservative': 'sum',
        'Allocated AV Aggressive': 'sum'}).reset_index()
    df_lien1_result['Conservative LTV (%)'] = df_lien1_result['Total outstanding debt as of 29.02.2024'] / df_lien1_result['Allocated AV Conservative'] * 100
    df_lien1_result['Aggressive LTV (%)'] = df_lien1_result['Total outstanding debt as of 29.02.2024'] / df_lien1_result['Allocated AV Aggressive'] * 100
    return df_lien1_result

def calculate_borrower_based_ltv(df_loans, df_collateral):
    loan_to_borrower = df_loans.set_index('Loan reference')['Borrower reference'].to_dict()
    df_collateral['Borrower reference'] = df_collateral['Loan reference'].map(loan_to_borrower)
    borrower_records = []

    for borrower in df_loans['Borrower reference'].unique():
        borrower_loans = df_loans[df_loans['Borrower reference'] == borrower]
        borrower_assets = df_collateral[df_collateral['Borrower reference'] == borrower]
        total_debt = borrower_loans['Total outstanding debt as of 29.02.2024'].sum()
        total_gav = borrower_assets['Gross Appraisal Value'].sum()
        ltv_total = (total_debt / total_gav * 100) if total_gav > 0 else 0
        borrower_records.append({
            'Borrower reference': borrower,
            'Borrower Loans': ', '.join(map(str, borrower_loans['Loan reference'].tolist())),
            'Total Debt': total_debt,
            'Total Appraisal Value': total_gav,
            'Total LTV (%)': ltv_total
        })
    return pd.DataFrame(borrower_records)

def calculate_loan_level_ltv_with_fallback(df_loans, df_collateral):
    # Logic already defined earlier — truncated here for brevity.
    pass

def calculate_comp_based_ltv(df_loans, df_collateral):
    df_merged = pd.merge(df_collateral, df_loans, on='Loan reference', how='left')
    G = nx.Graph()
    for _, row in df_merged.iterrows():
        G.add_edge('L_' + str(row['Loan reference']), 'A_' + str(row['Asset ID']))
    components = list(nx.connected_components(G))
    components_records = []
    for comp in components:
        loans = [node[2:] for node in comp if node.startswith('L_')]
        assets = [node[2:] for node in comp if node.startswith('A_')]
        comp_links = df_merged[df_merged['Loan reference'].isin(loans) & df_merged['Asset ID'].isin(assets)]
        total_gav = comp_links[['Asset ID', 'Gross Appraisal Value']].drop_duplicates()['Gross Appraisal Value'].sum()
        lien1_loans = comp_links[comp_links['Priority Ranking'] == 'Lien 1'].copy()
        gt1_loans = comp_links[comp_links['Priority Ranking'] != 'Lien 1'].copy()
        total_lien1_debt = lien1_loans[['Loan reference', 'Total outstanding debt as of 29.02.2024']].drop_duplicates()['Total outstanding debt as of 29.02.2024'].sum()
        total_lien_gt1_debt = gt1_loans[['Loan reference', 'Total outstanding debt as of 29.02.2024']].drop_duplicates()['Total outstanding debt as of 29.02.2024'].sum()
        adjusted_av_gt1 = total_gav - total_lien1_debt if total_gav > total_lien1_debt else 0
        ltv_total = (total_lien1_debt + total_lien_gt1_debt) / total_gav * 100 if total_gav else 0
        ltv_lien1 = total_lien1_debt / total_gav * 100 if total_gav else 0
        ltv_lien_gt1 = total_lien_gt1_debt / adjusted_av_gt1 * 100 if adjusted_av_gt1 else 0
        components_records.append({
            'Component Loans': ', '.join(loans),
            'Component Assets': ', '.join(assets),
            'Component Total outstanding debt as of 29.02.2024': total_lien1_debt + total_lien_gt1_debt,
            'Total Lien 1 Debt': total_lien1_debt,
            'Total Lien > 1 Debt': total_lien_gt1_debt,
            'Lien 1 LTV %': ltv_lien1,
            'Lien > 1 LTV %': ltv_lien_gt1,
            'Component Gross AV': total_gav,
            'Component LTV (%)': ltv_total
        })
    return pd.DataFrame(components_records)

def main():
    print("Select LTV Calculation Method:")
    print("1 - Loan-Level Conservative & Aggressive LTV")
    print("2 - Borrower-Level LTV")
    print("3 - Loan-Level LTV with Fallback")
    print("4 - Component-Based LTV (Network Graph)")
    choice = input("Enter the number corresponding to your choice: ")
    filepath = input("Enter the full path to the Excel data file: ")
    df_loans, df_collateral = load_clean_data(filepath)
    if choice == '1':
        result = calculate_loan_level_ltv(df_loans, df_collateral)
        result.to_excel("ltv_loan_level.xlsx", index=False)
        print("Loan-Level LTV exported to 'ltv_loan_level.xlsx'")
    elif choice == '2':
        result = calculate_borrower_based_ltv(df_loans, df_collateral)
        result.to_excel("ltv_borrower_level.xlsx", index=False)
        print("Borrower-Level LTV exported to 'ltv_borrower_level.xlsx'")
    elif choice == '3':
        result = calculate_loan_level_ltv_with_fallback(df_loans, df_collateral)
        result.to_excel("ltv_loan_with_fallback.xlsx", index=False)
        print("Loan-Level LTV with Fallback exported to 'ltv_loan_with_fallback.xlsx'")
    elif choice == '4':
        result = calculate_comp_based_ltv(df_loans, df_collateral)
        result.to_excel("ltv_component_based.xlsx", index=False)
        print("Component-Based LTV exported to 'ltv_component_based.xlsx'")
    else:
        print("Invalid choice. Please enter a number from 1 to 4.")

# Uncomment to run as script:
# if __name__ == "__main__":
#     main()
