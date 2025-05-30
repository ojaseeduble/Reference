
# FINAL LTV TOOL (v5): Dynamic Config + 4 Methods (Loan-Level, Borrower-Level, Component-Based, LP-Based)
import pandas as pd
import networkx as nx
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value

# === Read Configuration File ===
def load_config(config_path):
    df_config = pd.read_excel(config_path, sheet_name="Config")
    config_dict = dict(zip(df_config["Key"], df_config["Value"]))
    return config_dict

# === Load & Clean Data Using Config ===
def load_clean_data_from_config(config):
    file_path = config["datatape_path"]
    df_loans = pd.read_excel(file_path, sheet_name=config["loan_sheet"], header=None)
    df_collateral = pd.read_excel(file_path, sheet_name=config["collateral_sheet"], header=None)

    rows_to_drop = list(range(0, 4)) + [5, 6]
    df_loans = df_loans.drop(index=rows_to_drop).reset_index(drop=True)
    df_collateral = df_collateral.drop(index=rows_to_drop).reset_index(drop=True)

    df_loans.columns = df_loans.iloc[0]
    df_loans = df_loans.drop(index=0).reset_index(drop=True)
    df_collateral.columns = df_collateral.iloc[0]
    df_collateral = df_collateral.drop(index=0).reset_index(drop=True)

    # Subset and rename
    df_loans = df_loans[[config["loan_reference_col"], config["loan_amount_col"], config["borrower_ref_col"]]]
    df_collateral = df_collateral[[config["loan_reference_col"], config["collateral_ref_col"], config["plot_col"],
                                   config["gav_col"], config["priority_col"], config["collateral_type_col"], config["province_col"]]]

    df_collateral = df_collateral[df_collateral[config["gav_col"]] != 'n.a.']
    df_collateral['Asset ID'] = df_collateral[config["collateral_ref_col"]].astype(str) + '__' + df_collateral[config["plot_col"]].astype(str)

    df_loans[config["loan_amount_col"]] = pd.to_numeric(df_loans[config["loan_amount_col"]], errors='coerce')
    df_collateral[config["gav_col"]] = pd.to_numeric(df_collateral[config["gav_col"]], errors='coerce')

    return df_loans, df_collateral

# === LP-Based LTV Method ===
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

def calculate_lp_based_ltv(df_loans, df_collateral, config):
    loan_col = config["loan_reference_col"]
    debt_col = config["loan_amount_col"]

    G = nx.Graph()
    for idx, row in df_collateral.iterrows():
        G.add_edge("L_" + str(row[loan_col]), "A_" + str(row["Asset ID"]))

    components = list(nx.connected_components(G))
    results = []

    for comp in components:
        loans = [node[2:] for node in comp if node.startswith("L_")]
        assets = [node[2:] for node in comp if node.startswith("A_")]

        loans_lien1, loans_lien_gt1, collaterals_component = {}, {}, {}

        for loan in loans:
            exp_series = df_loans[df_loans[loan_col] == loan][debt_col]
            lien_series = df_collateral[df_collateral[loan_col] == loan][config["priority_col"]]
            if not exp_series.empty and not lien_series.empty:
                if lien_series.iloc[0] == "Lien 1":
                    loans_lien1[loan] = exp_series.iloc[0]
                else:
                    loans_lien_gt1[loan] = exp_series.iloc[0]

        for asset in assets:
            gav_series = df_collateral[df_collateral["Asset ID"] == asset][config["gav_col"]]
            if not gav_series.empty:
                collaterals_component[asset] = gav_series.iloc[0]

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

# === Main ===
def main():
    print("Using config-driven LTV tool...")
    config_path = input("Enter path to Excel config file: ").strip()
    config = load_config(config_path)

    df_loans, df_collateral = load_clean_data_from_config(config)

    print("Select LTV Calculation Method:")
    print("4 - LP-Based Allocation (Only LP logic is enabled in this version)")
    choice = input("Enter your choice (4): ").strip()

    if choice == "4":
        df_result = calculate_lp_based_ltv(df_loans, df_collateral, config)
        df_result.to_excel("ltv_lp_allocation_from_config.xlsx", index=False)
        print("LP-Based allocation exported to 'ltv_lp_allocation_from_config.xlsx'")
    else:
        print("Only Method 4 (LP) is implemented in this version. Extend other methods similarly.")

if __name__ == "__main__":
    main()
