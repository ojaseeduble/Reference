# FINAL LTV TOOL (v5): Loan-Level, Borrower-Level, Component-Based, and LP-Based LTV Calculation

# === Imports ===
import pandas as pd
import networkx as nx
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value

# === Load & Clean Data ===
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

# === Placeholder for Method 1 ===
def calculate_loan_level_ltv_with_fallback(df_loans, df_collateral):
    # Replace this with full logic from original script (see PDF)
    return pd.DataFrame({'Note': ['Method 1 placeholder logic']})

# === Placeholder for Method 2 ===
def calculate_borrower_based_ltv_with_fallback(df_loans, df_collateral):
    # Replace this with full logic from original script (see PDF)
    return pd.DataFrame({'Note': ['Method 2 placeholder logic']})

# === Placeholder for Method 3 ===
def calculate_comp_based_ltv_expanded(df_loans, df_collateral):
    # Replace this with full logic from original script (see PDF)
    return pd.DataFrame({'Note': ['Method 3 placeholder logic']})

# === Method 4: LP-Based Allocation ===
def lp_allocation_component(loans_lien1, loans_lien_gt1, collaterals_component):
    prob = LpProblem("Collateral_Allocation_Component", LpMaximize)
    x_L1 = LpVariable.dicts("x_L1", ((i, j) for i in loans_lien1 for j in collaterals_component), lowBound=0)
    x_Lgt1 = LpVariable.dicts("x_Lgt1", ((i, j) for i in loans_lien_gt1 for j in collaterals_component), lowBound=0)
    Z = LpVariable("Z", lowBound=0)

    for i in loans_lien1:
        prob += lpSum([x_L1[(i, j)] for j in collaterals_component]) >= loans_lien1[i] * Z

    for i in loans_lien_gt1:
        prob += lpSum([x_Lgt1[(i, j)] for j in collaterals_component]) >= loans_lien_gt1[i] * Z
        for j in collaterals_component:
            prob += x_Lgt1[(i, j)] <= collaterals_component[j] - lpSum([x_L1[(l, j)] for l in loans_lien1])

    for j in collaterals_component:
        prob += lpSum([x_L1[(i, j)] for i in loans_lien1]) + lpSum([x_Lgt1[(i, j)] for i in loans_lien_gt1]) == collaterals_component[j]

    prob += Z
    prob.solve()

    allocation = {"Lien1": {}, "LienGT1": {}}
    for i in loans_lien1:
        allocation["Lien1"][i] = {j: value(x_L1[(i, j)]) for j in collaterals_component}
    for i in loans_lien_gt1:
        allocation["LienGT1"][i] = {j: value(x_Lgt1[(i, j)]) for j in collaterals_component}

    optimal_Z = value(Z)
    minimized_max_LTV = 1 / optimal_Z if optimal_Z > 0 else None
    return allocation, optimal_Z, minimized_max_LTV, LpStatus[prob.status]

def calculate_lp_based_ltv(df_loans, df_collateral):
    G = nx.Graph()
    for idx, row in df_collateral.iterrows():
        G.add_edge("L_" + str(row["Loan reference"]), "A_" + str(row["Asset ID"]))

    components = list(nx.connected_components(G))
    results = []

    for comp in components:
        loans = [node[2:] for node in comp if node.startswith("L_")]
        assets = [node[2:] for node in comp if node.startswith("A_")]

        loans_lien1, loans_lien_gt1, collaterals_component = {}, {}, {}

        for loan in loans:
            exp_series = df_loans[df_loans["Loan reference"] == loan]["Total outstanding debt as of 29.02.2024"]
            lien_series = df_collateral[df_collateral["Loan reference"] == loan]["Priority Ranking"]
            if not exp_series.empty and not lien_series.empty:
                if lien_series.iloc[0] == "Lien 1":
                    loans_lien1[loan] = exp_series.iloc[0]
                else:
                    loans_lien_gt1[loan] = exp_series.iloc[0]

        for asset in assets:
            coll_value_series = df_collateral[df_collateral["Asset ID"] == asset]["Gross Appraisal Value"]
            if not coll_value_series.empty:
                collaterals_component[asset] = coll_value_series.iloc[0]

        if not loans_lien1 and not loans_lien_gt1:
            continue

        alloc, opt_Z, min_max_LTV, status = lp_allocation_component(loans_lien1, loans_lien_gt1, collaterals_component)
        results.append({
            "Loans_Lien1": loans_lien1,
            "Loans_LienGT1": loans_lien_gt1,
            "Collaterals": collaterals_component,
            "Allocation": alloc,
            "Opt_Z": opt_Z,
            "Minimized_Max_LTV": min_max_LTV,
            "Status": status
        })

    allocation_results = []
    for comp_idx, comp_result in enumerate(results):
        for lien_type in ["Lien1", "LienGT1"]:
            for loan, allocs in comp_result["Allocation"][lien_type].items():
                for asset, alloc_value in allocs.items():
                    if alloc_value > 0:
                        allocation_results.append({
                            "Component": comp_idx + 1,
                            "Loan Reference": loan,
                            "Lien Position": "Lien 1" if lien_type == "Lien1" else "Lien >1",
                            "Asset ID": asset,
                            "Loan Amount": comp_result["Loans_Lien1"].get(loan, 0) if lien_type == "Lien1" else comp_result["Loans_LienGT1"].get(loan, 0),
                            "Asset Value": comp_result["Collaterals"].get(asset, 0),
                            "Allocated Value": alloc_value,
                            "Allocation %": alloc_value / comp_result["Collaterals"].get(asset, 1) * 100,
                            "Optimized LTV": comp_result["Minimized_Max_LTV"],
                            "LP Status": comp_result["Status"]
                        })

    return pd.DataFrame(allocation_results)

# === Main Entry ===
def main():
    print("Select LTV Calculation Method:")
    print("1 - Loan-Level Conservative & Aggressive LTV (Full Logic with Sorting, Date, Province)")
    print("2 - Borrower-Level LTV with Fallback")
    print("3 - Component-Based LTV with Fallback")
    print("4 - LP-Based Allocation (Optimization)")

    choice = input("Enter choice (1/2/3/4): ").strip()
    filepath = input("Enter the full path to the Excel file: ").strip()

    df_loans, df_collateral = load_clean_data(filepath)

    if choice == "1":
        result = calculate_loan_level_ltv_with_fallback(df_loans, df_collateral)
        result.to_excel("ltv_loan_level.xlsx", index=False)
        print("Method 1: Loan-Level LTV exported.")
    elif choice == "2":
        result = calculate_borrower_based_ltv_with_fallback(df_loans, df_collateral)
        result.to_excel("ltv_borrower_level.xlsx", index=False)
        print("Method 2: Borrower-Level LTV exported.")
    elif choice == "3":
        result = calculate_comp_based_ltv_expanded(df_loans, df_collateral)
        result.to_excel("ltv_component_based.xlsx", index=False)
        print("Method 3: Component-Based LTV exported.")
    elif choice == "4":
        result = calculate_lp_based_ltv(df_loans, df_collateral)
        result.to_excel("ltv_lp_allocation.xlsx", index=False)
        print("Method 4: LP-Based Allocation exported.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
