
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value

def load_clean_data(filepath):
    """Load and clean loan and collateral data"""
    rows_to_drop = list(range(0, 4)) + [5, 6]

    # Load loan and collateral data
    df_loans = pd.read_excel(filepath, sheet_name='2.-Loan', header=None)
    df_collateral = pd.read_excel(filepath, sheet_name='4.-Loan & Collateral', header=None)

    # Drop unnecessary rows & reset index
    df_loans = df_loans.drop(index=rows_to_drop).reset_index(drop=True)
    df_collateral = df_collateral.drop(index=rows_to_drop).reset_index(drop=True)

    # Assign column headers
    df_loans.columns = df_loans.iloc[0]
    df_loans = df_loans.drop(index=0).reset_index(drop=True)

    df_collateral.columns = df_collateral.iloc[0]
    df_collateral = df_collateral.drop(index=0).reset_index(drop=True)

    # Select relevant columns
    df_loans = df_loans[['Loan reference', 'Total outstanding debt as of 29.02.2024', 'Borrower reference']]
    df_collateral = df_collateral[['Loan reference', 'Collateral unit reference', 'Plot',
                                   'Gross Appraisal Value', 'Priority Ranking', 'Collateral type', 'Province']]

    # Remove invalid Gross Appraisal Values
    df_collateral = df_collateral[df_collateral['Gross Appraisal Value'] != 'n.a.']

    # Assign Asset ID for unique collateral tracking
    df_collateral['Asset ID'] = df_collateral['Collateral unit reference'].astype(str) + '__' + df_collateral['Plot'].astype(str)

    # Convert loan amounts & appraisal values to numeric for optimization
    df_loans['Total outstanding debt as of 29.02.2024'] = pd.to_numeric(df_loans['Total outstanding debt as of 29.02.2024'], errors='coerce')
    df_collateral['Gross Appraisal Value'] = pd.to_numeric(df_collateral['Gross Appraisal Value'], errors='coerce')

    # Separate collateral into Lien 1 and Lien >1 for proper allocation
    df_lien1 = df_collateral[df_collateral['Priority Ranking'] == 'Lien 1'].copy()
    df_lien_gt1 = df_collateral[df_collateral['Priority Ranking'] != 'Lien 1'].copy()

    return df_loans, df_collateral, df_lien1, df_lien_gt1

def minimize_weighted_avg_LTV(df_loans, df_collateral, df_lien1, df_lien_gt1):
    """Set up LP model to minimize weighted average LTV"""
    loans = dict(zip(df_loans['Loan reference'], df_loans['Total outstanding debt as of 29.02.2024']))
    collaterals = dict(zip(df_collateral['Asset ID'], df_collateral['Gross Appraisal Value']))

    # Lien classifications
    collateral_lien1 = df_lien1.groupby('Loan reference')['Asset ID'].apply(list).to_dict()
    collateral_lien_gt1 = df_lien_gt1.groupby('Loan reference')['Asset ID'].apply(list).to_dict()

    # LP Problem Definition
    prob = LpProblem("Minimize_Weighted_Avg_LTV", LpMinimize)

    # Decision variables
    x = LpVariable.dicts("x", ((i, j) for i in loans for j in collaterals), lowBound=0)
    Z = LpVariable.dicts("Z", loans, lowBound=0)

    # Objective Function: Minimize weighted average LTV
    prob += lpSum([loans[i] * Z[i] for i in loans]), "Minimize_Weighted_LTV"

    # Constraint 1: LTV Calculation
    for i in loans:
        prob += lpSum([x[(i, j)] for j in collaterals]) * Z[i] >= loans[i], f"LTV_Calc_{i}"

    # Constraint 2: Full Collateral Allocation
    for j in collaterals:
        prob += lpSum([x[(i, j)] for i in loans]) == collaterals[j], f"Collateral_{j}_allocation"

    # Constraint 3: Lien Priority (Lien 1 first, then Lien >1)
    for i in collateral_lien_gt1:
        for j in collateral_lien_gt1[i]:
            prob += lpSum([x[(i, j)]]) <= collaterals[j] - lpSum([x[(l, j)] for l in collateral_lien1 if j in collateral_lien1.get(l, [])]), f"LienGT1_Constraint_{i}_{j}"

    # Solve LP
    prob.solve()

    # Extract results
    allocation = {i: {j: value(x[(i, j)]) for j in collaterals} for i in loans}
    optimal_LTVs = {i: value(Z[i]) for i in loans}
    weighted_avg_LTV = sum(loans[i] * optimal_LTVs[i] for i in loans) / sum(loans.values())

    return allocation, optimal_LTVs, weighted_avg_LTV, LpStatus[prob.status]

def save_results_to_excel(allocation, optimal_LTVs, weighted_avg_LTV, status, output_file="output.xlsx"):
    """Save results to an Excel file"""
    # Create DataFrames for Excel output
    df_alloc = pd.DataFrame([
        {'Loan': i, 'Asset': j, 'Allocated Collateral': alloc}
        for i, assets in allocation.items()
        for j, alloc in assets.items()
    ])

    df_LTVs = pd.DataFrame([
        {'Loan': i, 'Optimized LTV': optimal_LTVs[i]}
        for i in optimal_LTVs
    ])

    df_summary = pd.DataFrame({'LP Solver Status': [status], 'Minimized Weighted Average LTV': [weighted_avg_LTV]})

    # Save to Excel file
    with pd.ExcelWriter(output_file) as writer:
        df_alloc.to_excel(writer, sheet_name='Collateral Allocation', index=False)
        df_LTVs.to_excel(writer, sheet_name='Optimized LTVs', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Results saved to {output_file}")

def main():
    filepath = "path_to_your_excel_file.xlsx"  # Update with actual file path

    # Load data
    df_loans, df_collateral, df_lien1, df_lien_gt1 = load_clean_data(filepath)

    # Run LP optimization
    allocation, optimal_LTVs, weighted_avg_LTV, status = minimize_weighted_avg_LTV(df_loans, df_collateral, df_lien1, df_lien_gt1)

    # Save results to Excel
    save_results_to_excel(allocation, optimal_LTVs, weighted_avg_LTV, status)

if __name__ == "__main__":
    main()
