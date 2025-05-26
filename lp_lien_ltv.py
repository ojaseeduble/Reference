import pandas as pd
import networkx as nx
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value

# Load and clean the data
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

    # Build a unique Asset ID for collateral rows
    df_collateral['Asset ID'] = df_collateral['Collateral unit reference'].astype(str) + '__' + df_collateral['Plot'].astype(str)
    df_loans['Total outstanding debt as of 29.02.2024'] = pd.to_numeric(df_loans['Total outstanding debt as of 29.02.2024'], errors='coerce')
    df_collateral['Gross Appraisal Value'] = pd.to_numeric(df_collateral['Gross Appraisal Value'], errors='coerce')

    return df_loans, df_collateral

# LP model for optimizing collateral allocation
def lp_allocation_component(loans_lien1, loans_lien_gt1, collaterals_component):
    prob = LpProblem("Collateral_Allocation_Component", LpMaximize)

    # Decision variables: separate allocation for Lien 1 and Lien >1 loans
    x_L1 = LpVariable.dicts("x_L1", ((i, j) for i in loans_lien1 for j in collaterals_component), lowBound=0)
    x_Lgt1 = LpVariable.dicts("x_Lgt1", ((i, j) for i in loans_lien_gt1 for j in collaterals_component), lowBound=0)

    # Z is the inverse of the maximum LTV across loans
    Z = LpVariable("Z", lowBound=0)

    # Lien 1 loans get priority collateral allocation
    for i in loans_lien1:
        prob += lpSum([x_L1[(i, j)] for j in collaterals_component]) >= loans_lien1[i] * Z, f"Lien1_Loan_{i}_coverage"

    # Lien >1 loans receive only available collateral after Lien 1 allocation
    for i in loans_lien_gt1:
        prob += lpSum([x_Lgt1[(i, j)] for j in collaterals_component]) >= loans_lien_gt1[i] * Z, f"LienGT1_Loan_{i}_coverage"
        for j in collaterals_component:
            prob += x_Lgt1[(i, j)] <= collaterals_component[j] - lpSum([x_L1[(l, j)] for l in loans_lien1]), f"LienGT1_Constraint_{i}_{j}"

    # Collateral is fully allocated (Lien 1 + Lien >1 must sum to total collateral)
    for j in collaterals_component:
        prob += lpSum([x_L1[(i, j)] for i in loans_lien1]) + lpSum([x_Lgt1[(i, j)] for i in loans_lien_gt1]) == collaterals_component[j], f"Collateral_{j}_allocation"

    # Objective function: maximize Z (minimizing max LTV)
    prob += Z, "Maximize_Inverse_Max_LTV"

    # Solve LP
    prob.solve()

    # Extract results
    allocation = {"Lien1": {}, "LienGT1": {}}
    for i in loans_lien1:
        allocation["Lien1"][i] = {j: value(x_L1[(i, j)]) for j in collaterals_component}
    for i in loans_lien_gt1:
        allocation["LienGT1"][i] = {j: value(x_Lgt1[(i, j)]) for j in collaterals_component}

    optimal_Z = value(Z)
    minimized_max_LTV = 1 / optimal_Z if optimal_Z > 0 else None

    return allocation, optimal_Z, minimized_max_LTV, LpStatus[prob.status]

# Run LP across connected components
def run_lp_across_components(df_loans, df_collateral):
    G = nx.Graph()
    for idx, row in df_collateral.iterrows():
        G.add_edge("L_" + str(row["Loan reference"]), "A_" + str(row["Asset ID"]))

    components = list(nx.connected_components(G))
    results = []

    for comp in components:
        loans = [node[2:] for node in comp if node.startswith("L_")]
        assets = [node[2:] for node in comp if node.startswith("A_")]

        loans_lien1 = {}
        loans_lien_gt1 = {}
        collaterals_component = {}

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

    return results

# Main function to run everything
def main():
    filepath = "path_to_your_excel_file.xlsx"  # Update with actual file path

    # Load and clean the data
    df_loans, df_collateral = load_clean_data(filepath)

    # Run allocation LP across connected components
    results = run_lp_across_components(df_loans, df_collateral)

    # Print results
    for idx, comp_result in enumerate(results):
        print(f"Component {idx+1}:")
        print("Loans (Lien 1):", comp_result["Loans_Lien1"])
        print("Loans (Lien >1):", comp_result["Loans_LienGT1"])
        print("Collaterals:", comp_result["Collaterals"])
        print("Allocation:")
        for loan_type in ["Lien1", "LienGT1"]:
            for loan, allocs in comp_result["Allocation"][loan_type].items():
                print(f"  Loan {loan} ({loan_type}):")
                for asset, alloc_value in allocs.items():
                    print(f"    Asset {asset}: {alloc_value}")
        print("Optimized Inverse Max LTV (Z):", comp_result["Opt_Z"])
        print("Minimized Maximum LTV:", comp_result["Minimized_Max_LTV"])
        print("LP Solver Status:", comp_result["Status"])
        print("\n")

if __name__ == "__main__":
    main()
